from typing import Callable, Optional, Tuple

import cv2
import kornia
import numpy as np
import torch

from scripts.data import FilePath, resolve_model_path
from matchers.base import LocalFeatureMatcher
from matchers.config import OpenGlueConfig
from models.openglue.inference import load_openglue_matcher, match_og_superglue
from postprocesses.panet import PANetRefiner
from preprocesses.region import OverlapRegionCropper
from scripts.storage import LocalFeatureStorage, MatchingStorage


def read_image(path: str) -> np.ndarray:
    return cv2.imread(str(path))


class OpenGlueMatcher(LocalFeatureMatcher):
    def __init__(self, conf: OpenGlueConfig,
                 refiner: Optional[PANetRefiner] = None,
                 device: Optional[torch.device] = None):
        super().__init__(refiner=refiner)
        experiment_path = str(resolve_model_path(conf.experiment_path))
        self.conf = conf
        self.device = device
        self.matcher = load_openglue_matcher(
            experiment_path,
            conf.checkpoint_name,
            conf.descriptor_dim,
            device=device,
            resize_to=conf.resize_to,
            match_threshold=conf.match_threshold
        )

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

        #image_reader = image_reader or read_image
        #img1 = image_reader(str(path1))
        #img2 = image_reader(str(path2))

        inputs = build_openglue_input_data(
            kpts1, kpts2,
            descs1, descs2,
            shape1, shape2,
            scores1, scores2,
            device=self.device
        )

        idxs, mkpts1, mkpts2 = match_og_superglue(self.matcher, inputs)

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


def build_openglue_input_data(
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

    # Shape(1, N, dim)
    descriptors0 = torch.from_numpy(descs1).float().unsqueeze(0).to(device, non_blocking=True)
    descriptors1 = torch.from_numpy(descs2).float().unsqueeze(0).to(device, non_blocking=True)

    # Shape(1, N)
    responses0 = torch.from_numpy(scores1).float().unsqueeze(0).to(device, non_blocking=True)
    responses1 = torch.from_numpy(scores2).float().unsqueeze(0).to(device, non_blocking=True)

    # Shape(1, N, 2)
    keypoints0 = torch.from_numpy(kpts1).float().unsqueeze(0).to(device, non_blocking=True)
    keypoints1 = torch.from_numpy(kpts2).float().unsqueeze(0).to(device, non_blocking=True)

    lafs0 = torch.cat([
        torch.eye(2, device=device, dtype=keypoints0.dtype).unsqueeze(0).unsqueeze(1).expand(
            keypoints0.size(0), keypoints0.size(1), -1, -1
        ),
        keypoints0.unsqueeze(-1),
    ], dim=-1)
    lafs1 = torch.cat([
        torch.eye(2, device=device, dtype=keypoints1.dtype).unsqueeze(0).unsqueeze(1).expand(
            keypoints1.size(0), keypoints1.size(1), -1, -1
        ),
        keypoints1.unsqueeze(-1),
    ], dim=-1)

    data = {
        'h0': h0,
        'w0': w0,
        'h1': h1,
        'w1': w1,
        'lafs0': lafs0,
        'lafs1': lafs1,
        'descriptors0': descriptors0,
        'descriptors1': descriptors1,
        'responses0': responses0,
        'responses1': responses1
    }
    return data