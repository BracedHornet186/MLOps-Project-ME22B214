import pathlib
from typing import Optional, Union
from matplotlib.colors import to_rgb

import cv2
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F

from ns64_imc2022lib.config import MTLDescPipelineConfig, MTLDescModelConfig
from models.mtldesc.mtldesc import MTLDescExtractor


CONFIG_A = MTLDescPipelineConfig(
    model=MTLDescModelConfig(
        detection_threshold=0.02,
        nms_radius=4,
        border_remove=4
    ),
    to_rgb=True,
    extract_fn='extract_singlescale',
    top_k=2048
)


def read_image(
    path: str,
    to_rgb: bool = True
) -> np.ndarray:
    img = cv2.imread(str(path))
    if to_rgb:
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    return img


@torch.no_grad()
def extract_singlescale(
    net: MTLDescExtractor,
    img: np.ndarray,
    top_k: int = 10000,
) -> dict:
    assert not torch.backends.cudnn.benchmark

    shape = img.shape
    X, Y, S, C, Q, D = [], [], [], [], [], []

    res = net.predict(img=img)

    x = res['keypoints'][:,0]
    y = res['keypoints'][:,1]
    d = res['descriptors']
    scores = res['scores']

    X.append(x)
    Y.append(y)
    C.append(scores)
    D.append(d)
    Y = np.hstack(Y)
    X = np.hstack(X)
    scores = np.hstack(C)
    XY = np.stack([X, Y])
    XY = np.swapaxes(XY, 0, 1)
    D = np.vstack(D)
    idxs = scores.argsort()[-top_k or None:]
    predictions = {
        "keypoints": XY[idxs],
        "descriptors": D[idxs],
        "scores": scores[idxs],
        "shape": shape
    }
    return predictions


class MTLDescFeature(nn.Module):
    def __init__(self, weight_path: str):
        super().__init__()

        conf = CONFIG_A
        conf.model.weight_path = weight_path
        print(conf)

        self.conf = conf
        self.model = MTLDescExtractor(**conf.model.dict())

    def read_image(self, path: str) -> np.ndarray:
        return read_image(path, self.conf.to_rgb)

    def forward(self, image: np.ndarray, mask=None):
        """Compute keypoints, scores, descriptors for image"""
        pred = extract_singlescale(
            self.model,
            image,
            top_k=self.conf.top_k
        )
        keypoints = torch.from_numpy(pred['keypoints'])     # Shape(N, 2)
        descriptors = torch.from_numpy(pred['descriptors']) # Shape(N, 128)
        scores = torch.from_numpy(pred['scores'])           # Shape(N,)

        keypoints = keypoints.unsqueeze(0)          # Shape(1, N, 2)
        descriptors = descriptors.unsqueeze(0)      # Shape(1, N, 128)
        scores = scores.unsqueeze(0)                # Shape(1, N)

        lafs = torch.cat([
            torch.eye(2, device=keypoints.device, dtype=keypoints.dtype).unsqueeze(0).unsqueeze(1).expand(
                keypoints.size(0), keypoints.size(1), -1, -1
            ),
            keypoints.unsqueeze(-1),
        ], dim=-1)

        return lafs, scores, descriptors