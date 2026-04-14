import pathlib
from typing import Optional, Tuple, Union
from matplotlib.colors import to_rgb

import cv2
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F

from ns64_imc2022lib.config import PosFeatPipelineConfig, PosFeatModelConfig, PosFeatDetectorConfig
from ns64_imc2022lib.data import FilePath
from models.posfeat.extractor import PosFeatExtractor


CONFIG_A = PosFeatPipelineConfig(
    model='PosFeat',
    model_config=PosFeatModelConfig(),
    detector='generate_kpts_single',
    detector_config=PosFeatDetectorConfig(
        num_pts=2048,
        stable=True,
        nms_radius=3,
        thr=0.5
    ),
)


def load_model(
    conf: PosFeatPipelineConfig
) -> PosFeatExtractor:
    return PosFeatExtractor(conf)


def read_image(
    path: FilePath,
    extractor: PosFeatExtractor,
    image_size: Optional[int] = None
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    img_tensor, img_orig, scale = extractor.read_image(path, image_size=image_size)
    return img_tensor, img_orig, scale


@torch.no_grad()
def extract(
    model: PosFeatExtractor,
    img1: torch.Tensor,
    scale1: np.ndarray,
) -> dict:
    inputs1 = {'im1': img1, 'scale': scale1}
    pred1 = model.extract(inputs1)
    return pred1


class PoSFeatFeature(nn.Module):
    def __init__(self, load_path: str):
        super().__init__()

        conf = CONFIG_A
        conf.load_path = load_path
        print(conf)

        self.conf = conf
        self.model = load_model(conf)

    def read_image(self, path: str) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        return read_image(path, self.model)

    def forward(self, img: torch.Tensor, scale: np.ndarray, mask=None):
        """Compute keypoints, scores, descriptors for image"""
        pred = extract(self.model, img, scale)

        keypoints = torch.from_numpy(pred['kpt'])   # Shape(N, 2)
        descriptors = pred['desc']                  # Shape(1, N, 128)
        scores = pred['kp_score']                   # Shape(1, N)

        keypoints = keypoints.unsqueeze(0)          # Shape(1, N, 2)

        lafs = torch.cat([
            torch.eye(2, device=keypoints.device, dtype=keypoints.dtype).unsqueeze(0).unsqueeze(1).expand(
                keypoints.size(0), keypoints.size(1), -1, -1
            ),
            keypoints.unsqueeze(-1),
        ], dim=-1)

        return lafs, scores, descriptors