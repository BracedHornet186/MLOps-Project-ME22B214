from typing import Callable, Optional

import cv2
import kornia
import numpy as np
import torch
import torchvision.transforms as T
from kornia.utils.image import image_to_tensor

from data import FilePath, resolve_model_path
from features.base import (
    LocalFeatureHandler,
    LocalFeatureOutputs,
    postprocess,
    read_image,
)
from features.config import DELFPytorchConfig
from models.delf_pytorch.delf import Delf_V1
from models.delf_pytorch.extractor import FeatureExtractor
from preprocess import resize_image_tensor
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.orientation import OrientationNormalizer
from preprocesses.region import Cropper


# PIL, RGB order, ToTensor()
class DELFPytorchFeatHandler(LocalFeatureHandler):
    def __init__(
        self, conf: DELFPytorchConfig, device: Optional[torch.device] = None
    ):
        self.conf = conf
        self.device = device
        checkpoints_path = str(resolve_model_path(conf.weight_path))
        model = FeatureExtractor(conf.get_extractor_config(checkpoints_path))
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
        assert resize

        ori_normalizer = None
        if orientation is not None:
            ori_normalizer = OrientationNormalizer(degree=orientation)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if cropper:
            cropper.set_original_image(img)
            cropped_img = cropper.crop_ndarray_image_gray(img)
        else:
            cropped_img = img

        if ori_normalizer:
            raise NotImplementedError

        rotator = None
        if rotation:
            raise NotImplementedError

        x = T.ToTensor()(cropped_img)
        x = self.preprocess(x, resize=resize)
        x = x.to(self.device, non_blocking=True)[None]

        keypoints, scores, descriptors = self.model.__extract_delf_feature__(x)

        kpts = keypoints.reshape(-1, 2).to(self.device)
        kpts = kpts[:, [1, 0]]
        scores = scores.reshape(-1)
        descs = descriptors.reshape(len(kpts), -1)

        if rotator:
            raise NotImplementedError

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]  # Shape(1, N, 2, 3) -> Shape(N, 2, 3)
        outputs = (lafs, scores, descs)
        outputs = postprocess(outputs, cropped_img, x, cropper=cropper)
        return outputs

    @torch.inference_mode()
    def extract_by_keypoints(
        self,
        path: FilePath,
        pre_sampled_keypoints: np.ndarray,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image,
    ) -> LocalFeatureOutputs:
        raise NotImplementedError

    def to_torch_image(self, img: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    def preprocess(
        self, x: torch.Tensor, resize: Optional[ResizeConfig] = None
    ) -> torch.Tensor:
        if resize is not None:
            x = resize_image_tensor(x, resize)
        return x