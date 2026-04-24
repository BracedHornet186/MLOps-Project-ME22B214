
from typing import Callable, Optional

import cv2
import numpy as np
import kornia
import torch
from pydantic import BaseModel

from scripts.data import FilePath, resolve_model_path
from features.base import LocalFeatureHandler, LocalFeatureOutputs, lafs_to_keypoints, read_image
from features.config import FeatureBoosterConfig
from features.alike import ALIKEHandler
from models.featurebooster.model import FeatureBooster
from scripts.preprocess import resize_image_opencv, resize_image_tensor
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper
from scripts.workspace import log


# Adapted from https://github.com/SJTU-ViSYS/FeatureBooster/blob/main/config.yaml
BASE_CONFIGS = {
    'ALIKE+Boost-F': {
        'keypoint_dim': 3,
        'keypoint_encoder': [32, 64, 128, 128],
        'descriptor_encoder': [256, 128],
        'descriptor_dim': 128,
        'Attentional_layers': 9,
        'last_activation': None,
        'l2_normalization': True,
        'output_dim': 128,
    },
    'ALIKE+Boost-B': {
        'keypoint_dim': 3,
        'keypoint_encoder': [32, 64, 128, 128],
        'descriptor_encoder': [256, 128],
        'descriptor_dim': 128,
        'Attentional_layers': 9,
        'last_activation': 'tanh',
        'l2_normalization': False,
        'output_dim': 256,
    }
}


class FeatureBoosterHandler(LocalFeatureHandler):
    def __init__(self,
                 conf: FeatureBoosterConfig,
                 device: Optional[torch.device] = None):
        self.conf = conf
        self.device = device

        if conf.descriptor == 'alike':
            assert conf.alike
            feature = ALIKEHandler(conf.alike, device=device)
        else:
            raise ValueError(conf.descriptor)
        
        weight_path = resolve_model_path(conf.weight_path)
        log(f'[FeatureBooster] Loading weights from {weight_path}')
        weight = torch.load(weight_path, map_location='cpu')
        feature_booster = FeatureBooster(BASE_CONFIGS[conf.model_type])
        feature_booster.load_state_dict(weight)

        self.feature = feature
        self.feature_booster = feature_booster.eval().to(self.device)

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
        if rotation:
            raise NotImplementedError

        if isinstance(self.feature, ALIKEHandler):
            lafs, scores, descs = self.feature(path, resize=resize)
        else:
            raise NotImplementedError
        
        kpts = lafs_to_keypoints(lafs.clone())
        kpts = normalize_keypoints_torch(kpts, img.shape)
        kpts = torch.cat([kpts, scores[:, None]], dim=1)

        out = self.feature_booster(descs, kpts)
        if 'boost-b' in self.conf.model_type.lower():
            out = (out >= 0).cpu().detach().numpy()
            descriptors = np.packbits(out, axis=1, bitorder='little')
            descriptors = torch.from_numpy(descriptors)
        else:
            descriptors = out
        
        return lafs, scores, descriptors


def normalize_keypoints(keypoints: np.ndarray, image_shape: tuple) -> np.ndarray:
    """
    Args
    ----
    keypoints : np.ndarray
        Shape(N, 2), xy order

    image_shape : Tuple[int, ...]
        A tuple of (height, width) or (height, width, channel)
    """
    x0 = image_shape[1] / 2
    y0 = image_shape[0] / 2
    scale = max(image_shape) * 0.7
    kps = np.array(keypoints)
    kps[:, 0] = (keypoints[:, 0] - x0) / scale
    kps[:, 1] = (keypoints[:, 1] - y0) / scale
    return kps 


def normalize_keypoints_torch(keypoints: torch.Tensor, image_shape: tuple) -> torch.Tensor:
    """
    Args
    ----
    keypoints : np.ndarray
        Shape(N, 2), xy order

    image_shape : Tuple[int, ...]
        A tuple of (height, width) or (height, width, channel)
    """
    x0 = image_shape[1] / 2
    y0 = image_shape[0] / 2
    scale = max(image_shape) * 0.7
    keypoints[:, 0] = (keypoints[:, 0] - x0) / scale
    keypoints[:, 1] = (keypoints[:, 1] - y0) / scale
    return keypoints 