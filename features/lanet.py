from typing import Callable, Optional

import cv2
import kornia
import numpy as np
import torch
import torchvision.transforms as T
from pydantic import BaseModel

from data import FilePath, resolve_model_path
from features.base import (LocalFeatureHandler, LocalFeatureOutputs,
                           postprocess, read_image)
from features.config import LANetConfig
from models.lanet.network_v1.model import PointModel as LANetV1
from preprocess import resize_image_opencv, resize_image_tensor
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class LANetHandler(LocalFeatureHandler):
    def __init__(self,
                 conf: LANetConfig,
                 device: Optional[torch.device] = None):
        self.conf = conf
        self.device = device
        if conf.model_version == 'v0':
            raise NotImplementedError
        elif conf.model_version == 'v1':
            model = LANetV1(is_test=True)
        else:
            raise ValueError(conf.model_version)
        
        if conf.weight_path:
            weight_path = resolve_model_path(conf.weight_path)
            weight = torch.load(weight_path, map_location='cpu')
            model.load_state_dict(weight['model_state'])

        self.model = model.eval().to(self.device)

    @torch.inference_mode()
    def __call__(
        self,
        path: FilePath,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image
    ) -> LocalFeatureOutputs:
        img = image_reader(str(path))
        # NOTE:
        # https://github.com/wangch-g/lanet/blob/master/datasets/hp_loader.py#L74-L76
        # LANet requires BGR order
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize:
            resized_img, scale, mask = resize_image_opencv(img, conf=resize)
        else:
            resized_img = img
        
        if rotation:
            raise NotImplementedError
        
        x = self.to_torch_image(resized_img)
        outputs = self.extract(x)
        outputs = postprocess(
            outputs, img,
            kornia.utils.image_to_tensor(resized_img, keepdim=False)
        )
        return outputs

    def to_torch_image(self, img: np.ndarray) -> torch.Tensor:
        return T.ToTensor()(img)
    
    def preprocess(
        self,
        x: torch.Tensor,
        resize: Optional[ResizeConfig] = None
    ) -> torch.Tensor:
        raise NotImplementedError
    
    def extract(self, x: torch.Tensor, *args, **kwargs) -> LocalFeatureOutputs:
        x = x.unsqueeze(0)
        x = x.to(self.device)

        scores, coords, descs = self.model(x)
        B, _, Hc, Wc = descs.shape

        scores = torch.cat([coords, scores], dim=1).view(3, -1).t().cpu().numpy()
        descs = descs.view(256, Hc, Wc).view(256, -1).t().cpu().numpy()

        keeps = scores[:, 2] > self.conf.threshold
        descs = descs[keeps, :]
        scores = scores[keeps, :]

        if self.conf.topk is not None:
            keeps = (-scores[:, 2]).argsort()[:self.conf.topk]
            scores = scores[keeps, :]
            descs = descs[keeps, :]

        kpts = torch.from_numpy(scores[:, :2]).to(self.device, non_blocking=True)
        scores = torch.from_numpy(scores[:, 2]).to(self.device, non_blocking=True)
        descs = torch.from_numpy(descs).to(self.device, non_blocking=True)

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]

        return lafs, scores, descs
