from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn

from data import FilePath, resolve_model_path
from matchers.base import LocalFeatureMatcher
from matchers.config import GIMLightGlueConfig
from postprocesses.panet import PANetRefiner
from preprocesses.region import OverlapRegionCropper
from storage import LocalFeatureStorage, MatchingStorage
from models.gim.gluefactory.models.matchers.lightglue import LightGlue


def read_image(path: str) -> np.ndarray:
    return cv2.imread(str(path))


class GIMLightGlueMatcher(LocalFeatureMatcher):
    def __init__(self, conf: GIMLightGlueConfig,
                 refiner: Optional[PANetRefiner] = None,
                 device: Optional[torch.device] = None):
        super().__init__(refiner=refiner)
        self.conf = conf
        self.device = device
        model = LightGlue({
            'filter_threshold': 0.1,
            'flash': False,
            'checkpointed': True,
        })
        checkpoints_path = str(resolve_model_path(conf.weight_path))
        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('superpoint.'):
                state_dict.pop(k)
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        model.load_state_dict(state_dict)
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
        image_reader: Optional[Callable] = None
    ) -> np.ndarray:
        lafs1, kpts1, scores1, descs1 = feature_storage.get(path1)
        lafs2, kpts2, scores2, descs2 = feature_storage.get(path2)

        inputs = build_lightglue_input_data(
            kpts1, kpts2,
            descs1, descs2,
            shape1, shape2,
            scores1, scores2,
            device=self.device
        )

        preds = self.model(inputs)
        idxs = preds["matches"][0]
        idxs = idxs.cpu().numpy()

        if self.use_overlap_region_cropper and cropper:
            idxs = self.filter_matches_out_of_overlap_region(
                idxs, kpts1, kpts2, cropper
            )

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


def build_lightglue_input_data(
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    descs1: np.ndarray,
    descs2: np.ndarray,
    shape1: Tuple[int, int],
    shape2: Tuple[int, int],
    scores1: Optional[np.ndarray] = None,
    scores2: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None
) -> dict:
    assert scores1 is not None and scores2 is not None
    h0, w0, *_ = shape1
    h1, w1, *_ = shape2

    # Shape(N, dim)
    descriptors0 = torch.from_numpy(descs1).float().unsqueeze(0).to(device, non_blocking=True)
    descriptors1 = torch.from_numpy(descs2).float().unsqueeze(0).to(device, non_blocking=True)

    # Shape(N,)
    # responses0 = torch.from_numpy(scores1).float().to(device, non_blocking=True)
    # responses1 = torch.from_numpy(scores2).float().to(device, non_blocking=True)

    # Shape(1, N, 2)
    keypoints0 = torch.from_numpy(kpts1).float().unsqueeze(0).to(device, non_blocking=True)
    keypoints1 = torch.from_numpy(kpts2).float().unsqueeze(0).to(device, non_blocking=True)

    data = {
        'descriptors0': descriptors0,
        'descriptors1': descriptors1,
        'keypoints0': keypoints0,
        'keypoints1': keypoints1
    }
    return data