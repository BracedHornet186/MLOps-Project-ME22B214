from __future__ import annotations

from collections.abc import Callable

import cv2
import dad
import kornia.feature
import numpy as np
import torch
from dad.utils import check_not_i16, sample_keypoints
from PIL import Image

from scripts.data import FilePath, resolve_model_path
from features.base import (
    LocalFeatureHandler,
    LocalFeatureOutputs,
    keypoints_to_lafs,
    lafs_to_keypoints,
    postprocess,
    read_image,
)
from features.config import EdgePoint2Config
from models.edgepoint2.edgepoint2 import EdgePoint2Wrapper
from scripts.preprocess import resize_image_opencv
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class EdgePoint2Handler(LocalFeatureHandler):
    def __init__(self, conf: EdgePoint2Config, device: torch.device | None = None):
        self.conf = conf
        self.device = device
        self.model = (
            EdgePoint2Wrapper(
                conf.model_type,
                str(resolve_model_path(conf.weight_path)),
                top_k=conf.top_k,
                score=conf.score,
            )
            .eval()
            .to(device)
        )

    def __call__(
        self,
        path: FilePath,
        resize: ResizeConfig | None = None,
        rotation: RotationConfig | None = None,
        cropper: Cropper | None = None,
        orientation: int | None = None,
        image_reader: Callable = read_image,
    ) -> LocalFeatureOutputs:
        if rotation:
            raise NotImplementedError
        img = image_reader(str(path))
        H, W, _ = img.shape

        if cropper:
            cropper.set_original_image(img)
            img = cropper.crop_ndarray_image(img)

        if orientation is not None:
            raise NotImplementedError

        if rotation:
            raise NotImplementedError

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize is not None:
            img, scale, mask = resize_image_opencv(img, conf=resize, order3ch="hwc")
        x = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        x = x.to(self.device, non_blocking=True)

        with torch.inference_mode():
            result = self.model(x)

        kpts = result[0]["keypoints"]
        descs = result[0]["descriptors"]
        scores = result[0]["scores"]

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]  # Shape(1, N, 2, 3) -> Shape(N, 2, 3)

        outputs = postprocess((lafs, scores, descs), img, x, cropper=cropper)
        return outputs
