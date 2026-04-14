from typing import Callable, Optional

import kornia
import torch
import torch.nn as nn
from lightglue.sift import SIFT
from lightglue.utils import load_image, numpy_image_to_torch

from data import FilePath, resolve_model_path
from features.base import (
    LocalFeatureHandler,
    LocalFeatureOutputs,
    keypoints_to_lafs,
    lafs_to_keypoints,
    read_image,
)
from features.config import LightGlueSIFTConfig
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.orientation import OrientationNormalizer
from preprocesses.region import Cropper


class LightGlueSIFTHandler(LocalFeatureHandler):
    def __init__(
        self, conf: LightGlueSIFTConfig, device: Optional[torch.device] = None
    ):
        self.conf = conf
        self.device = device
        model = SIFT(max_num_keypoints=conf.max_num_keypoints).eval().to(device)
        self.model = model

    @torch.inference_mode()
    def __call__(
        self,
        path: FilePath,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image,
    ) -> LocalFeatureOutputs:
        img = image_reader(str(path))

        ori_normalizer = None
        if orientation is not None:
            ori_normalizer = OrientationNormalizer(degree=orientation)

        img = img[..., ::-1]    # BGR->RGB
        img = numpy_image_to_torch(img)
        img = img.to(self.device, non_blocking=True)

        if cropper:
            cropper.set_original_image(img)
            img = cropper.crop_tensor_image(img)

        if ori_normalizer:
            ori_normalizer.set_original_image(img)
            img = ori_normalizer.get_upright_image_tensor()

        rotator = None
        if rotation:
            raise NotImplementedError

        if resize:
            raise NotImplementedError
        else:
            preds = self.model.extract(img)

        kpts = preds["keypoints"].reshape(-1, 2)
        scores = preds["keypoint_scores"].reshape(-1)
        descs = preds["descriptors"].reshape(len(kpts), -1)

        if rotator:
            raise NotImplementedError

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]  # Shape(1, N, 2, 3) -> Shape(N, 2, 3)

        if ori_normalizer:
            kpts = lafs_to_keypoints(lafs)
            kpts = ori_normalizer.keypoints_to_original_coords_torch(kpts)
            lafs = keypoints_to_lafs(kpts)

        if cropper:
            kpts = lafs_to_keypoints(lafs)
            kpts = cropper.convert_cropped_to_original_coordinates_torch(kpts)
            lafs = keypoints_to_lafs(kpts)

        outputs = (lafs, scores, descs)
        return outputs
