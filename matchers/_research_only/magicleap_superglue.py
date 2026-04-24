from pathlib import Path
from typing import Optional, Callable, Tuple

import cv2
import kornia
import numpy as np
import torch

from scripts.data import FilePath, resolve_model_path
from matchers.base import LocalFeatureMatcher
from matchers.config import MagicLeapSuperGlueConfig
from postprocesses.panet import PANetRefiner
from models._research_only.magicleap.superglue import SuperGlue
from models._research_only.magicleap.utils import frame2tensor
from preprocesses.region import OverlapRegionCropper
from scripts.storage import InMemoryLocalFeatureStorage, LocalFeatureStorage, MatchingStorage


def read_image(path: str) -> np.ndarray:
    return cv2.imread(str(path))


class MagicLeapSuperGlueMatcher(LocalFeatureMatcher):
    def __init__(self, conf: MagicLeapSuperGlueConfig,
                 refiner: Optional[PANetRefiner] = None,
                 device: Optional[torch.device] = None):
        super().__init__(refiner=refiner)
        weight_path = str(resolve_model_path(conf.weight_path))
        model = SuperGlue({
            'weights': conf.weights,
            'sinkhorn_iterations': conf.sinkhorn_iterations,
            'match_threshold': conf.match_threshold,
            'model_path': weight_path
        })

        self.conf = conf
        self.device = device
        self.model = model.eval().to(device)

    @property
    def min_matches(self) -> Optional[int]:
        return self.conf.min_matches
    
    @property
    def use_overlap_region_cropper(self) -> bool:
        return self.conf.use_overlap_region_cropper

    @torch.inference_mode()
    def __call__(
        self,
        path1: FilePath,
        path2: FilePath,
        shape1: Tuple[int, int],    # (H, W)
        shape2: Tuple[int, int],    # (H, W)
        feature_storage: LocalFeatureStorage,
        matching_storage: Optional[MatchingStorage] = None,
        cropper: Optional[OverlapRegionCropper] = None,
        image_reader: Callable = read_image
    ) -> np.ndarray:
        lafs1, kpts1, scores1, descs1 = feature_storage.get(path1)
        lafs2, kpts2, scores2, descs2 = feature_storage.get(path2)

        descs1 = descs1.T   # Shape(N, dim) -> Shape(dim, N)
        descs2 = descs2.T   # Shape(N, dim) -> Shape(dim, N)

        #img1 = image_reader(str(path1))
        #img2 = image_reader(str(path2))
        #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        #x1 = frame2tensor(img1, self.device)
        #x2 = frame2tensor(img2, self.device)

        data = {
            'keypoints0': torch.from_numpy(kpts1).float().unsqueeze(0).to(self.device),
            'keypoints1': torch.from_numpy(kpts2).float().unsqueeze(0).to(self.device),
            'descriptors0': torch.from_numpy(descs1).float().unsqueeze(0).to(self.device),
            'descriptors1': torch.from_numpy(descs2).float().unsqueeze(0).to(self.device),
            'scores0': torch.from_numpy(scores1).float().unsqueeze(0).to(self.device),
            'scores1': torch.from_numpy(scores2).float().unsqueeze(0).to(self.device),
            'shape0': (1, 1, *shape1),
            'shape1': (1, 1, *shape2),
            #'image0': x1,
            #'image1': x2
        }

        preds = self.model(data)
        preds = {k: v[0].detach().cpu().numpy()
                 for k, v in preds.items()}

        matches = preds['matches0']
        scores = preds['matching_scores0']

        valid = matches > -1
        #mkpts1 = data['keypoints0'][valid]
        #mkpts2 = data['keypoints1'][matches[valid]]
        #mscores = scores[valid]

        idxs = np.concatenate([
            np.where(valid)[0][None].T,
            matches[valid][None].T
        ], axis=1)      # Shape(#matches, 2)

        if self.use_overlap_region_cropper and cropper:
            idxs = self.filter_matches_out_of_overlap_region(
                idxs, kpts1, kpts2, cropper
            )

        if self.refiner:
            img1 = image_reader(str(path1))
            img2 = image_reader(str(path2))
            new_kpts1, new_kpts2 = self.refiner.refine_matched_keypoints(
                img1, img2, kpts1, kpts2, idxs
            )
            # TODO
            assert isinstance(feature_storage, InMemoryLocalFeatureStorage)
            feature_storage.keypoints[Path(path1).name] = new_kpts1.copy()
            feature_storage.keypoints[Path(path2).name] = new_kpts2.copy()

        if matching_storage:
            if self.min_matches is None or len(idxs) >= self.min_matches:
                matching_storage.add(path1, path2, idxs)
        return idxs

    def match(
        self,
        descs1: np.ndarray,
        descs2: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

