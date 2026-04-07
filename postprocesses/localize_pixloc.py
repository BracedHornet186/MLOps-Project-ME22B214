import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pycolmap
from omegaconf import DictConfig
from omegaconf import OmegaConf as oc
import torch
from tqdm import tqdm
from PIL import Image, ExifTags
import models.pixloc.utils.colmap

from data import resolve_model_path
from models.pixloc.localization.feature_extractor import FeatureExtractor
from models.pixloc.localization.model3d import Model3D
from models.pixloc.localization.refiners import RetrievalRefiner, BaseRefiner
from models.pixloc.localization.tracker import BaseTracker
from models.pixloc.pixlib.geometry import Camera
from models.pixloc.pixlib.geometry.wrappers import Pose
from models.pixloc.pixlib.models import get_model
from models.pixloc.pixlib.utils.experiments import load_experiment
from models.pixloc.utils.quaternions import qvec2rotmat, rotmat2qvec
from pipelines.scene import Scene
from storage import InMemoryKeypointStorage, InMemoryMatchingStorage


default_confs = {
    'from_retrieval': {
        'experiment': resolve_model_path('PIXLOC_MEGADEPTH'),
        'features': {},
        'optimizer': {
            'num_iters': 100,
            'pad': 2,  # to 1?
        },
        'refinement': {
            'num_dbs': 5,
            'multiscale': [4, 1],
            'point_selection': 'all',
            'normalize_descriptors': True,
            'average_observations': True,   # False
            'filter_covisibility': False,
            'do_pose_approximation': False,
        },
    },
}


def localize_pixloc(
    reference_sfm: pycolmap.Reconstruction,
    no_registered_query_full_keys: List[str],
    scene: Scene,
):
    """
    NOTE: Downloading: "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth" to /home/ns64/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth
    """
    conf = default_confs['from_retrieval']
    device = torch.device('cuda:0')
    localizer = CustomRetrievalLocalizer(
        reference_sfm, conf, no_registered_query_full_keys, scene, device
    )
    outputs, _ = localizer.run_batched()
    return outputs


def get_focal(image_path: str, err_on_default: bool = False) -> float:
    image = Image.open(image_path)
    max_size = max(image.size)

    exif = image.getexif()
    focal = None
    if exif is not None:
        focal_35mm = None
        # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/util/bitmap.cc#L299
        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == 'FocalLengthIn35mmFilm':
                focal_35mm = float(value)
                break

        if focal_35mm is not None:
            focal = focal_35mm / 35. * max_size
    
    if focal is None:
        if err_on_default:
            raise RuntimeError("Failed to find focal length")

        # failed to find it in exif, use prior
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size

    return focal


def read_image(path, grayscale=False):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f'Could not read image at {path}.')
    if not grayscale:
        image = image[..., ::-1]
    return image


class CustomLocalizer:
    def __init__(
        self,
        reference_sfm: pycolmap.Reconstruction,
        conf: Union[DictConfig, Dict],
        no_registered_query_full_keys: List[str],
        scene: Scene,
        device: Optional[torch.device]
    ):
        device = device or torch.device('cpu')

        queries = {}
        for qkey in no_registered_query_full_keys:
            qidx = scene.full_key_to_idx(qkey)
            qpath = str(scene.image_paths[qidx])
            height, width = scene.get_image_shape(qpath)
            focal = get_focal(qpath)
            param_arr = np.array([focal, width / 2, height / 2, 0.1])
            #qcam = pycolmap.Camera(
            #    'SIMPLE_RADIAL', int(width), int(height), param_arr
            #)
            qcam = models.pixloc.utils.colmap.Camera(
                None, 'SIMPLE_RADIAL',int(width), int(height), param_arr
            )
            qname = str(Path(qpath).name)
            queries[qname] = qcam

        # Loading feature extractor and optimizer from experiment or scratch
        conf = oc.create(conf)
        conf_features = conf.features.get('conf', {})
        conf_optim = conf.get('optimizer', {})
        if conf.get('experiment'):
            pipeline = load_experiment(
                    conf.experiment,
                    {'extractor': conf_features, 'optimizer': conf_optim})
            pipeline = pipeline.to(device)
            print(
                'Use full pipeline from experiment %s with config:\n%s',
                conf.experiment, oc.to_yaml(pipeline.conf))
            extractor = pipeline.extractor
            optimizer = pipeline.optimizer
            if isinstance(optimizer, torch.nn.ModuleList):
                optimizer = list(optimizer)
        else:
            assert 'name' in conf.features
            extractor = get_model(conf.features.name)(conf_features)
            optimizer = get_model(conf.optimizer.name)(conf_optim)

        self.reference_sfm = reference_sfm
        self.no_registered_query_full_keys = no_registered_query_full_keys
        self.scene = scene
        self.model3d = Model3D(reference_sfm)
        self.queries = queries
        self.conf = conf
        self.device = device
        self.optimizer = optimizer
        self.extractor = FeatureExtractor(
            extractor, device, conf.features.get('preprocessing', {})
        )

    def run_query(self, name: str, camera: Camera):
        raise NotImplementedError

    def run_batched(self, skip: Optional[int] = None,
                    ) -> Tuple[Dict[str, Tuple], Dict]:
        output_poses = {}
        output_logs = {
            'configuration': oc.to_yaml(self.conf),
            'localization': {},
        }

        print('Starting the localization process...')
        query_names = list(self.queries.keys())[::skip or 1]
        for name in tqdm(query_names):
            camera = Camera.from_colmap(self.queries[name])
            try:
                ret = self.run_query(name, camera)
                if ret is None:
                    continue
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print('Out of memory')
                    torch.cuda.empty_cache()
                    ret = {'success': False}
                else:
                    continue
            output_logs['localization'][name] = ret
            if ret['success']:
                R, tvec = ret['T_refined'].numpy()
            elif 'T_init' in ret:
                R, tvec = ret['T_init'].numpy()
            else:
                continue
            #output_poses[name] = (rotmat2qvec(R), tvec)

            key1 = self.scene.image_name_to_full_key(name)
            output_poses[key1] = {
                'R': R.copy(),
                't': tvec.copy()
            }

        return output_poses, output_logs


class CustomRetrievalLocalizer(CustomLocalizer):
    def __init__(
        self,
        reference_sfm: pycolmap.Reconstruction,
        conf: Union[DictConfig, Dict],
        no_registered_query_full_keys: List[str],
        scene: Scene,
        device: Optional[torch.device]
    ):
        super().__init__(reference_sfm,
                         conf,
                         no_registered_query_full_keys,
                         scene,
                         device)

        global_descriptors = None

        self.refiner = CustomRetrievalRefiner(
            self.device, self.optimizer, self.model3d, self.extractor, scene,
            self.conf.refinement, global_descriptors=global_descriptors)

        self.retrieval = scene.get_retrieval_dict(key_type='short_key')

    def run_query(self, name: str, camera: Camera):
        dbs = []
        for r in self.retrieval[name]:
            if r in self.model3d.name2id:
                dbs.append(self.model3d.name2id[r])

        if len(dbs) == 0:
            # TODO
            # dbs = [_id for _, _id in self.model3d.name2id.items()][:5]
            return None

        #dbs = [self.model3d.name2id[r] for r in self.retrieval[name]]
        loc = None
        ret = self.refiner.refine(name, camera, dbs, loc=loc)
        return ret


class CustomBaseRefiner(BaseRefiner):
    base_default_config = dict(
        layer_indices=None,
        min_matches_db=10,
        num_dbs=1,
        min_track_length=3,
        min_points_opt=10,
        point_selection='all',
        average_observations=False,
        normalize_descriptors=True,
        compute_uncertainty=True,
    )

    default_config = dict()
    tracker: BaseTracker = None

    def __init__(self,
                 device: torch.device,
                 optimizer: torch.nn.Module,
                 model3d: Model3D,
                 feature_extractor: FeatureExtractor,
                 scene: Scene,
                 conf: Union[DictConfig, Dict]):
        self.device = device
        self.optimizer = optimizer
        self.model3d = model3d
        self.feature_extractor = feature_extractor
        self.scene = scene

        self.conf = oc.merge(
            oc.create(self.base_default_config),
            oc.create(self.default_config),
            oc.create(conf))

    def refine_pose_using_features(self,
                                   features_query: List[torch.tensor],
                                   scales_query: List[float],
                                   qcamera: Camera,
                                   T_init: Pose,
                                   features_p3d: List[List[torch.Tensor]],
                                   p3dids: List[int]) -> Dict:
        """Perform the pose refinement using given dense query feature-map.
        """
        # decompose descriptors and uncertainities, normalize descriptors
        weights_ref = []
        features_ref = []
        for level in range(len(features_p3d[0])):
            feats = torch.stack([feat[level] for feat in features_p3d], dim=0)
            feats = feats.to(self.device)
            if self.conf.compute_uncertainty:
                feats, weight = feats[:, :-1], feats[:, -1:]
                weights_ref.append(weight)
            if self.conf.normalize_descriptors:
                feats = torch.nn.functional.normalize(feats, dim=1)
            assert not feats.requires_grad
            features_ref.append(feats)

        # query dense features decomposition and normalization
        features_query = [feat.to(self.device) for feat in features_query]
        if self.conf.compute_uncertainty:
            weights_query = [feat[-1:] for feat in features_query]
            features_query = [feat[:-1] for feat in features_query]
        if self.conf.normalize_descriptors:
            features_query = [torch.nn.functional.normalize(feat, dim=0)
                              for feat in features_query]

        p3d = np.stack([self.model3d.points3D[p3did].xyz for p3did in p3dids])

        T_i = T_init
        ret = {'T_init': T_init}
        # We will start with the low res feature map first
        for idx, level in enumerate(reversed(range(len(features_query)))):
            F_q, F_ref = features_query[level], features_ref[level]
            qcamera_feat = qcamera.scale(scales_query[level])

            if self.conf.compute_uncertainty:
                W_ref_query = (weights_ref[level], weights_query[level])
            else:
                W_ref_query = None

            print(f'Optimizing at level {level}.')
            opt = self.optimizer
            if isinstance(opt, (tuple, list)):
                if self.conf.layer_indices:
                    opt = opt[self.conf.layer_indices[level]]
                else:
                    opt = opt[level]
            T_opt, fail = opt.run(p3d, F_ref, F_q, T_i.to(F_q),
                                  qcamera_feat.to(F_q),
                                  W_ref_query=W_ref_query)

            self.log_optim(i=idx, T_opt=T_opt, fail=fail, level=level,
                           p3d=p3d, p3d_ids=p3dids,
                           T_init=T_init, camera=qcamera_feat)
            if fail:
                return {**ret, 'success': False}
            T_i = T_opt

        # Compute relative pose w.r.t. initilization
        T_opt = T_opt.cpu().double()
        dR, dt = (T_init.inv() @ T_opt).magnitude()
        return {
            **ret,
            'success': True,
            'T_refined': T_opt,
            'diff_R': dR.item(),
            'diff_t': dt.item(),
        }
    
    def read_image(self, name: str) -> np.ndarray:
        img = self.scene.get_image(
            self.scene.image_paths[self.scene.short_key_to_idx(name)]
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def refine_query_pose(self, qname: str, qcamera: Camera, T_init: Pose,
                          p3did_to_dbids: Dict[int, List],
                          multiscales: Optional[List[int]] = None) -> Dict:

        dbid_to_p3dids = self.model3d.get_dbid_to_p3dids(p3did_to_dbids)
        if multiscales is None:
            multiscales = [1]

        rnames = [self.model3d.dbs[i].name for i in dbid_to_p3dids.keys()]
        images_ref = [self.read_image(name) for name in rnames]

        for image_scale in multiscales:
            # Compute the reference observations
            # TODO: can we compute this offline before hand?
            dbid_p3did_to_feats = dict()
            for idx, dbid in enumerate(dbid_to_p3dids):
                p3dids = dbid_to_p3dids[dbid]

                features_ref_dense, scales_ref = self.dense_feature_extraction(
                        images_ref[idx], rnames[idx], image_scale)
                dbid_p3did_to_feats[dbid] = self.interp_sparse_observations(
                        features_ref_dense, scales_ref, dbid, p3dids)
                del features_ref_dense

            p3did_to_feat = self.aggregate_features(
                    p3did_to_dbids, dbid_p3did_to_feats)
            if self.conf.average_observations:
                p3dids = list(p3did_to_feat.keys())
                p3did_to_feat = [p3did_to_feat[p3did] for p3did in p3dids]
            else:  # duplicate the observations
                p3dids, p3did_to_feat = list(zip(*[
                    (p3did, feat) for p3did, feats in p3did_to_feat.items()
                    for feat in zip(*feats)]))

            # Compute dense query feature maps
            image_query = self.read_image(qname)
            features_query, scales_query = self.dense_feature_extraction(
                        image_query, qname, image_scale)

            ret = self.refine_pose_using_features(features_query, scales_query,
                                                  qcamera, T_init,
                                                  p3did_to_feat, p3dids)
            if not ret['success']:
                print(f"Optimization failed for query {qname}")
                break
            else:
                T_init = ret['T_refined']
        return ret


class CustomRetrievalRefiner(CustomBaseRefiner):
    default_config = dict(
        multiscale=None,
        filter_covisibility=False,
        do_pose_approximation=False,
        do_inlier_ranking=False,
    )

    def __init__(self, *args, **kwargs):
        self.global_descriptors = kwargs.pop('global_descriptors', None)
        super().__init__(*args, **kwargs)

    def refine(self, qname: str, qcamera: Camera, dbids: List[int],
               loc: Optional[Dict] = None) -> Dict:

        if self.conf.do_inlier_ranking:
            assert loc is not None

        if self.conf.do_inlier_ranking and loc['PnP_ret']['success']:
            inliers = loc['PnP_ret']['inliers']
            ninl_dbs = self.model3d.get_db_inliers(loc, dbids, inliers)
            dbids = self.model3d.rerank_and_filter_db_images(
                    dbids, ninl_dbs, self.conf.num_dbs,
                    self.conf.min_matches_db)
        else:
            assert self.conf.point_selection == 'all'
            dbids = dbids[:self.conf.num_dbs]
            if self.conf.do_pose_approximation or self.conf.filter_covisibility:
                dbids = self.model3d.covisbility_filtering(dbids)
            inliers = None

        if self.conf.do_pose_approximation:
            if self.global_descriptors is None:
                raise RuntimeError(
                        'Pose approximation requires global descriptors')
            Rt_init = self.model3d.pose_approximation(
                    qname, dbids, self.global_descriptors)
        else:
            id_init = dbids[0]
            image_init = self.model3d.dbs[id_init]
            Rt_init = (image_init.qvec2rotmat(), image_init.tvec)
        T_init = Pose.from_Rt(*Rt_init)
        fail = {'success': False, 'T_init': T_init, 'dbids': dbids}

        p3did_to_dbids = self.model3d.get_p3did_to_dbids(
                dbids, loc, inliers, self.conf.point_selection,
                self.conf.min_track_length)

        # Abort if there are not enough 3D points after filtering
        if len(p3did_to_dbids) < self.conf.min_points_opt:
            print("Not enough valid 3D points to optimize")
            return fail

        ret = self.refine_query_pose(qname, qcamera, T_init, p3did_to_dbids,
                                     self.conf.multiscale)
        ret = {**ret, 'dbids': dbids}
        return ret