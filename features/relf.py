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
    postprocess,
    read_image,
)
from features.config import RELFConfig
from models.relf.descriptor_utils import DescGroupPoolandNorm
from models.relf.load_model import load_model
from preprocess import resize_image_tensor
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class RELFHandler(LocalFeatureHandler):
    def __init__(
        self,
        conf: RELFConfig,
        detector: LocalFeatureExtractor,
        device: Optional[torch.device] = None,
    ):
        os.environ['Orientation'] = str(conf.descriptor.num_group)

        self.conf = conf
        self.device = device
        self.detector = detector
        self.descriptor = RELFDescriptor(conf).eval().to(device)
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(
        self,
        path: FilePath,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image,
    ) -> LocalFeatureOutputs:
        if rotation:
            raise NotImplementedError

        detector_outputs = self.detector(path, cropper=cropper, rotation=rotation)
        _, kpts, scores, _ = detector_outputs
        kpts = torch.from_numpy(kpts)
        scores = torch.from_numpy(scores)

        img = image_reader(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.transform(img)
        x = self.preprocess(x, resize=resize)

        origin_h, origin_w, *_ = img.shape
        resize_h, resize_w = x.shape[-2:]

        kpts[:, 0] *= (resize_w / origin_w)
        kpts[:, 1] *= (resize_h / origin_h)

        outputs = self.extract(x, kpts, scores)
        outputs = postprocess(outputs, img, x)
        return outputs

    def to_torch_image(self, img: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    def preprocess(
        self, x: torch.Tensor, resize: Optional[ResizeConfig] = None
    ) -> torch.Tensor:
        if resize is not None:
            x = resize_image_tensor(x, resize)
        return x

    def extract(
        self, x: torch.Tensor, kpts: torch.Tensor, scores: torch.Tensor, *args, **kwargs
    ) -> LocalFeatureOutputs:
        x = x.to(self.device)
        kpts = kpts.to(self.device)

        kpts, descs = self.descriptor(x.unsqueeze(0), kpts.unsqueeze(0).float())
        kpts = kpts[0]
        descs = descs[0]

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]
        return lafs, scores, descs


class RELFDescriptor(nn.Module):
    def __init__(self, conf: RELFConfig):
        super().__init__()
        args = conf.descriptor.to_args(str(resolve_model_path(conf.weight_path)))
        self.model = load_model(args)
        self.pool_and_norm = DescGroupPoolandNorm(args)

    def __call__(
        self, image: torch.Tensor, kpts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = image.size(0)
        assert batch_size == 1

        descs = self.model(image, kpts)

        ## kpts torch.tensor ([B, K, 2]), desc torch.tensor ([B, K, CG])
        k1, d1 = self.pool_and_norm.desc_pool_and_norm_infer(kpts, descs)
        if isinstance(k1, list):
            assert len(k1) == 1
            k1 = k1[0].unsqueeze(0)
        if isinstance(d1, list):
            assert len(d1) == 1
            d1 = d1[0].unsqueeze(0)

        assert isinstance(k1, torch.Tensor)
        assert isinstance(d1, torch.Tensor)
        return k1, d1
