import os
from typing import Callable, Optional

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from data import FilePath, resolve_model_path
from extractor import LocalFeatureExtractor
from features.base import (
    LocalFeatureHandler,
    LocalFeatureOutputs,
    keypoints_to_lafs,
    lafs_to_keypoints,
    postprocess,
    read_image,
)
from lightglue.utils import numpy_image_to_torch
from features.config import XFeatConfig
from models.xfeat.xfeat import XFeat
from preprocess import resize_image_tensor
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class XFeatHandler(LocalFeatureHandler):
    def __init__(
        self,
        conf: XFeatConfig,
        device: Optional[torch.device] = None,
    ):
        self.conf = conf
        self.device = device
        self.model = XFeat(weights=str(resolve_model_path(conf.weight_path)))

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if cropper:
            cropper.set_original_image(img)
            img = cropper.crop_ndarray_image(img)

        x = numpy_image_to_torch(img)
        x = x.to(self.device, non_blocking=True)
        x = x[None]
        if resize:
            x = self.preprocess(x, resize=resize)

        rotator = None
        if rotation:
            raise NotImplementedError

        preds = self.model.detectAndCompute(x, top_k=self.conf.topk)

        kpts = preds[0]["keypoints"]
        scores = preds[0]["scores"]
        descs = preds[0]["descriptors"]

        if rotator:
            raise NotImplementedError

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]  # Shape(1, N, 2, 3) -> Shape(N, 2, 3)

        outputs = (lafs, scores, descs)
        outputs = postprocess(outputs, img, x, cropper=cropper)

        return outputs

    def to_torch_image(self, img: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    def preprocess(
        self, x: torch.Tensor, resize: Optional[ResizeConfig] = None
    ) -> torch.Tensor:
        if resize is not None:
            x = resize_image_tensor(x, resize)
        return x
