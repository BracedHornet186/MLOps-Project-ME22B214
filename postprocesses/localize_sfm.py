import pickle
import copy
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pycolmap
from PIL import ExifTags, Image
from tqdm import tqdm

from models.pixloc.utils.quaternions import qvec2rotmat
from pipelines.scene import Scene
from storage import InMemoryKeypointStorage, InMemoryMatchingStorage


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
            if ExifTags.TAGS.get(tag, None) == "FocalLengthIn35mmFilm":
                focal_35mm = float(value)
                break

        if focal_35mm is not None:
            focal = focal_35mm / 35.0 * max_size

    if focal is None:
        if err_on_default:
            raise RuntimeError("Failed to find focal length")

        # failed to find it in exif, use prior
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size

    return focal


def do_covisibility_clustering(
    frame_ids: List[int], reconstruction: pycolmap.Reconstruction
):
    clusters = []
    visited = set()
    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        clusters.append([])
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            observed = reconstruction.images[exploration_frame].points2D
            connected_frames = {
                obs.image_id
                for p2D in observed
                if p2D.has_point3D()
                for obs in reconstruction.points3D[p2D.point3D_id].track.elements
            }
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


class QueryLocalizer:
    def __init__(self, reconstruction, config=None):
        self.reconstruction = reconstruction
        self.config = config or {}

    def localize(self, points2D_all, points2D_idxs, points3D_id, query_camera):
        points2D = points2D_all[points2D_idxs]
        points3D = [self.reconstruction.points3D[j].xyz for j in points3D_id]
        if len(points3D) == 0:
            points3D = np.empty((0, 3))
        ret = pycolmap.absolute_pose_estimation(
            points2D,
            points3D,
            query_camera,
            estimation_options=self.config.get("estimation", {}),
            refinement_options=self.config.get("refinement", {}),
        )
        return ret


def pose_from_cluster(
    localizer: QueryLocalizer,
    qname: str,
    query_camera: pycolmap.Camera,
    db_ids: List[int],
    scene: Scene,
    keypoint_storage: InMemoryKeypointStorage,
    matching_storage: InMemoryMatchingStorage,
    **kwargs,
):
    kpq = keypoint_storage.get(qname)
    kpq += 0.5  # COLMAP coordinates

    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches = 0
    for i, db_id in enumerate(db_ids):
        image = localizer.reconstruction.images[db_id]
        if image.num_points3D == 0:
            print(f"No 3D points found for {image.name}.")
            continue
        points3D_ids = np.array(
            [p.point3D_id if p.has_point3D() else -1 for p in image.points2D]
        )

        try:
            matches = matching_storage.get_unique(qname, image.name, scene)
        except KeyError:
            continue
        # matches, _ = get_matches(matches_path, qname, image.name)

        matches = matches[points3D_ids[matches[:, 1]] != -1]
        num_matches += len(matches)
        for idx, m in matches:
            id_3D = points3D_ids[m]
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            # avoid duplicate observations
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

    idxs = list(kp_idx_to_3D.keys())
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
    ret = localizer.localize(kpq, mkp_idxs, mp3d_ids, query_camera, **kwargs)
    # NOTE:
    # {'cam_from_world': Rigid3d(quat_xyzw=[0.214981, 0.0127556, -0.0128597, 0.97645], t=[-0.185633, 11.3493, -33.0228]), 'num_inliers': 3, 'inliers': array([ True,  True,  True])}

    if ret is None:
        ret = {"success": False}
    else:
        ret["camera"] = {
            "model": query_camera.model,
            "width": query_camera.width,
            "height": query_camera.height,
            "params": query_camera.params,
        }
        ret["success"] = True

    # mostly for logging and post-processing
    mkp_to_3D_to_db = [
        (j, kp_idx_to_3D_to_db[i][j]) for i in idxs for j in kp_idx_to_3D[i]
    ]
    log = {
        "db": db_ids,
        "PnP_ret": ret,
        "keypoints_query": kpq[mkp_idxs],
        "points3D_ids": mp3d_ids,
        "points3D_xyz": None,  # we don't log xyz anymore because of file size
        "num_matches": num_matches,
        "keypoint_index_to_db": (mkp_idxs, mkp_to_3D_to_db),
    }
    return ret, log


def localize_sfm(
    reference_sfm: pycolmap.Reconstruction,
    no_registered_query_full_keys: List[str],
    scene: Scene,
    keypoint_storage: InMemoryKeypointStorage,
    matching_storage: InMemoryMatchingStorage,
    ransac_thresh: float = 4.0,
    #ransac_thresh: float = 12.0,
    covisibility_clustering: bool = False,
    config: Dict = None,
):
    db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}
    config = {"estimation": {"ransac": {"max_error": ransac_thresh}}, **(config or {})}
    localizer = QueryLocalizer(reference_sfm, config)

    cam_from_world = {}
    retrieval_dict = scene.get_retrieval_dict(key_type="short_key")
    # qnames = [scene.idx_to_key(i, 'short_key')
    #          for i in range(len(scene.image_paths))]
    # queries = []
    # for i, qname in enumerate(qnames, start=1):
    #     if i not in reference_sfm.cameras:
    #         continue
    #     camera = reference_sfm.cameras[i]
    #     queries.append((str(qname), reference_sfm.cameras[i]))
    qnames = []
    queries = []
    for qkey in no_registered_query_full_keys:
        qidx = scene.output_key_to_idx(qkey)
        qpath = str(scene.image_paths[qidx])
        height, width = scene.get_image_shape(qpath)
        focal = get_focal(qpath)
        param_arr = np.array([focal, width / 2, height / 2, 0.1])
        qcam = pycolmap.Camera(
            model="SIMPLE_RADIAL",
            width=int(width),
            height=int(height),
            params=param_arr,
        )
        qname = str(Path(qpath).name)
        qnames.append(qname)
        queries.append((qname, qcam))

    for qname, qcam in tqdm(queries):
        if qname not in retrieval_dict:
            print(f"No images retrieved for query image {qname}. Skipping...")
            continue
        db_names = retrieval_dict[qname]
        db_ids = []
        for n in db_names:
            if n not in db_name_to_id:
                print(f"(q={qname}) Image {n} was retrieved but not in database")
                continue
            print(f"(q={qname}) Image {n} was found in database")
            db_ids.append(db_name_to_id[n])

        if len(db_ids) == 0:
            print(f"(q={qname}) All retrieved images are not registered. Skip ...")
            continue

        if covisibility_clustering:
            clusters = do_covisibility_clustering(db_ids, reference_sfm)
            best_inliers = 0
            best_cluster = None
            logs_clusters = []
            for i, cluster_ids in enumerate(clusters):
                ret, log = pose_from_cluster(
                    localizer, qname, qcam, cluster_ids, features, matches
                )
                if ret["success"] and ret["num_inliers"] > best_inliers:
                    best_cluster = i
                    best_inliers = ret["num_inliers"]
                logs_clusters.append(log)
            if best_cluster is not None:
                ret = logs_clusters[best_cluster]["PnP_ret"]
                poses[qname] = (ret["qvec"], ret["tvec"])
            # logs['loc'][qname] = {
            #    'db': db_ids,
            #    'best_cluster': best_cluster,
            #    'log_clusters': logs_clusters,
            #    'covisibility_clustering': covisibility_clustering,
            # }
        else:
            ret, log = pose_from_cluster(
                localizer,
                qname,
                qcam,
                db_ids,
                scene,
                keypoint_storage,
                matching_storage,
            )
            if ret["success"]:
                cam_from_world[qname] = ret["cam_from_world"]
            else:
                closest = reference_sfm.images[db_ids[0]]
                cam_from_world[qname] = closest.cam_from_world
            log["covisibility_clustering"] = covisibility_clustering
            # print(log)

    print(f"Localized {len(cam_from_world)} / {len(queries)} images.")
    outputs = {}
    for qname, t in cam_from_world.items():
        # world to cam
        #qvec = t.rotation.quat[[3, 0, 1, 2]]
        #tvec = t.translation
        key1 = scene.data_schema.format_output_key(scene.dataset, scene.scene, qname)
        # rotmat = R.from_quat(qvec.copy()).as_matrix().copy()

        #rotmat = qvec2rotmat(qvec.copy())
        outputs[key1] = {
            "R": copy.deepcopy(t.rotation.matrix()),
            "t": copy.deepcopy(np.array(t.translation))
        }

    return outputs


def main(
    reference_sfm: Union[Path, pycolmap.Reconstruction],
    queries: Path,
    retrieval: Path,
    features: Path,
    matches: Path,
    results: Path,
    ransac_thresh: int = 12,
    covisibility_clustering: bool = False,
    prepend_camera_name: bool = False,
    config: Dict = None,
):
    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    queries = parse_image_lists(queries, with_intrinsics=True)
    retrieval_dict = parse_retrieval(retrieval)

    logger.info("Reading the 3D model...")
    if not isinstance(reference_sfm, pycolmap.Reconstruction):
        reference_sfm = pycolmap.Reconstruction(reference_sfm)
    db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}

    config = {"estimation": {"ransac": {"max_error": ransac_thresh}}, **(config or {})}
    localizer = QueryLocalizer(reference_sfm, config)

    poses = {}
    logs = {
        "features": features,
        "matches": matches,
        "retrieval": retrieval,
        "loc": {},
    }
    logger.info("Starting localization...")
    for qname, qcam in tqdm(queries):
        if qname not in retrieval_dict:
            logger.warning(f"No images retrieved for query image {qname}. Skipping...")
            continue
        db_names = retrieval_dict[qname]
        db_ids = []
        for n in db_names:
            if n not in db_name_to_id:
                logger.warning(f"Image {n} was retrieved but not in database")
                continue
            db_ids.append(db_name_to_id[n])

        if covisibility_clustering:
            clusters = do_covisibility_clustering(db_ids, reference_sfm)
            best_inliers = 0
            best_cluster = None
            logs_clusters = []
            for i, cluster_ids in enumerate(clusters):
                ret, log = pose_from_cluster(
                    localizer, qname, qcam, cluster_ids, features, matches
                )
                if ret["success"] and ret["num_inliers"] > best_inliers:
                    best_cluster = i
                    best_inliers = ret["num_inliers"]
                logs_clusters.append(log)
            if best_cluster is not None:
                ret = logs_clusters[best_cluster]["PnP_ret"]
                poses[qname] = (ret["qvec"], ret["tvec"])
            logs["loc"][qname] = {
                "db": db_ids,
                "best_cluster": best_cluster,
                "log_clusters": logs_clusters,
                "covisibility_clustering": covisibility_clustering,
            }
        else:
            ret, log = pose_from_cluster(
                localizer, qname, qcam, db_ids, features, matches
            )
            if ret["success"]:
                poses[qname] = (ret["qvec"], ret["tvec"])
            else:
                closest = reference_sfm.images[db_ids[0]]
                poses[qname] = (closest.qvec, closest.tvec)
            log["covisibility_clustering"] = covisibility_clustering
            logs["loc"][qname] = log

    logger.info(f"Localized {len(poses)} / {len(queries)} images.")
    logger.info(f"Writing poses to {results}...")
    with open(results, "w") as f:
        for q in poses:
            qvec, tvec = poses[q]
            qvec = " ".join(map(str, qvec))
            tvec = " ".join(map(str, tvec))
            name = q.split("/")[-1]
            if prepend_camera_name:
                name = q.split("/")[-2] + "/" + name
            f.write(f"{name} {qvec} {tvec}\n")

    logs_path = f"{results}_logs.pkl"
    logger.info(f"Writing logs to {logs_path}...")
    with open(logs_path, "wb") as f:
        pickle.dump(logs, f)
    logger.info("Done!")
