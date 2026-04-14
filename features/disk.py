from typing import Callable, Optional

import cv2
import kornia
import numpy as np
import torch
from kornia.feature import DISK
from kornia.utils.image import image_to_tensor
from pydantic import BaseModel

from data import FilePath, resolve_model_path
from features.base import (LocalFeatureHandler, LocalFeatureOutputs,
                           postprocess, read_image)
from features.config import DISKConfig
from preprocess import resize_image_tensor
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class DISKHandler(LocalFeatureHandler):
    def __init__(self,
                 conf: DISKConfig,
                 device: Optional[torch.device] = None):
        self.conf = conf
        self.device = device
        disk = DISK()
        if conf.weight_path:
            state_dict = torch.load(
                resolve_model_path(conf.weight_path),
                map_location='cpu'
            )
            disk.load_state_dict(state_dict['extractor'])
        self.feature = disk.eval().to(device)

    def __call__(
        self,
        path: FilePath,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image
    ) -> LocalFeatureOutputs:
        if rotation:
            raise NotImplementedError
        img = image_reader(str(path))
        x = self.to_torch_image(img)
        x = self.preprocess(x, resize=resize)
        outputs = self.extract(x)
        outputs = postprocess(outputs, img, x)
        return outputs

    def to_torch_image(self, img: np.ndarray) -> torch.Tensor:
        x = image_to_tensor(img, False).float() / 255.
        x = kornia.color.bgr_to_rgb(x)
        return x
    
    def preprocess(
        self,
        x: torch.Tensor,
        resize: Optional[ResizeConfig] = None
    ) -> torch.Tensor:
        if resize is not None:
            x = resize_image_tensor(x, resize)
        return x
    
    def extract(self, x: torch.Tensor, *args, **kwargs) -> LocalFeatureOutputs:
        x = x.to(self.device)

        outputs = self.feature(x, self.conf.num_features, pad_if_not_divisible=True)[0]
        kpts: torch.Tensor = outputs.keypoints
        scores: torch.Tensor = outputs.detection_scores
        descs: torch.Tensor = outputs.descriptors

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]
        return lafs, scores, descs
