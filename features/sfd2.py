from typing import Callable, Optional, Tuple

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel

from scripts.data import FilePath, resolve_model_path
from features.base import (LocalFeatureHandler, LocalFeatureOutputs, create_rotator,
                           postprocess, read_image)
from features.config import SFD2Config
from models.sfd2.extractor import extract_resnet_return
from models.sfd2.sfd2 import ResSegNet, ResSegNetV2
from scripts.preprocess import resize_image_opencv, resize_image_tensor
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class SFD2Handler(LocalFeatureHandler):
    def __init__(self,
                 conf: SFD2Config,
                 device: Optional[torch.device] = None):
        weight_path = str(resolve_model_path(conf.weight_path))
        model, extractor = get_model(
            conf.model_name,
            weight_path,
            use_stability=conf.use_stability
        )
        self.conf = conf
        self.device = device
        self.model = model.eval().cuda()
        self.extractor = extractor

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
        if resize:
            resized_img, scale, mask = resize_image_opencv(img, conf=resize)
        else:
            resized_img = img
            scale = None
        
        #resized_img = resized_img[None]     # Shape(H, W) -> Shape(1, H, W)

        x = torch.from_numpy(resized_img).unsqueeze(0).to(self.device, non_blocking=True)
        x = x.float() / 255.
        x = x.permute((0, 3, 1, 2))

        rotator = None
        if rotation:
            rotator = create_rotator(rotation, x.shape[-2], x.shape[-1], self.device)
            x = rotator.transform_homography_variants_tensor(x)
            assert x.shape[0] == 1
        
        preds = self.extractor(
            self.model,
            img=x,
            mask=None,
            topK=self.conf.max_keypoints,
            conf_th=self.conf.conf_th,
            scales=self.conf.scales
        )

        kpts = torch.from_numpy(preds['keypoints']).float().to(self.device, non_blocking=True)
        scores = torch.from_numpy(preds['scores']).float().to(self.device, non_blocking=True)
        descs = torch.from_numpy(preds['descriptors']).float().to(self.device, non_blocking=True)

        if rotator:
            kpts = rotator.inverse_transform_keypoints_tensor(kpts)

        if scale is not None:
            kpts[:, 0] = (kpts[:, 0] + 0.5) * scale[0] - 0.5  # x
            kpts[:, 1] = (kpts[:, 1] + 0.5) * scale[1] - 0.5  # y

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]    # Shape(1, N, 2, 3) -> Shape(N, 2, 3)

        outputs = (lafs, scores, descs)
        return outputs
    
    def read_image(self, path: FilePath, *args, **kwargs) -> np.ndarray:
        #return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return cv2.imread(str(path))
    

def get_model(model_name: str,
              weight_path: FilePath,
              use_stability: bool = False) -> Tuple[nn.Module, Callable]:
    if model_name == 'ressegnet':
        model = ResSegNet(outdim=128, require_stability=use_stability).eval()
        model.load_state_dict(torch.load(weight_path)['model'], strict=True)
        extractor = extract_resnet_return
    elif model_name == 'ressegnetv2':
        model = ResSegNetV2(outdim=128, require_stability=use_stability).eval()
        model.load_state_dict(torch.load(weight_path)['model'], strict=False)
        extractor = extract_resnet_return
    else:
        raise ValueError(model_name)

    return model, extractor