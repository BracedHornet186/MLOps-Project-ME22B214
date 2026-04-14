"""Adapted from https://github.com/LSXI7/MINIMA/blob/main/data_engine/tools/depth/depth_transfer.py"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2


def init_depth_model(
    weight_path: str | Path,
    device: torch.device,
    encoder: str = "vitl",
) -> DepthAnythingV2:
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }

    # encoder = 'vitl'  # or 'vits', 'vitb', 'vitg'
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(
        torch.load(
            str(weight_path),
            map_location="cpu",
        )
    )
    model = model.to(device).eval()
    return model


@torch.inference_mode()
def predict_depth(
    image: np.ndarray, model: DepthAnythingV2, device: torch.device
) -> np.ndarray:
    depth = model.infer_image(image, device)  # HxW raw depth map in numpy
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")

    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)

    depth_colored = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    return depth_colored  # BGR?


@torch.inference_mode()
def depth_transfer_single(
    img_path: str | Path, model: DepthAnythingV2, device: torch.device
) -> np.ndarray:
    image = cv2.imread(str(img_path))
    processed_image = predict_depth(image, model, device)
    return processed_image
