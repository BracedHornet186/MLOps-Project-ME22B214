
from pathlib import Path
from typing import Callable, Optional, Tuple, Any

import cv2
import kornia
import numpy as np
import torch
from kornia.utils.image import image_to_tensor
from PIL import Image

from scripts.data import FilePath, LocalFeatureOutputs
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.homography_adaptation import HomographyAdaptation
from preprocesses.region import Cropper


def read_image(path: str) -> np.ndarray:
    return cv2.imread(str(path))


def read_image_pil2cv2(path: str) -> np.ndarray:
    img = np.array(Image.open(str(path)).convert('RGB'))
    img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    return img
    

def create_rotator(conf: RotationConfig,
                   height: int, width: int,
                   device: Optional[torch.device] = None) -> HomographyAdaptation:
    assert len(conf.angles) == 1
    rotator = HomographyAdaptation(height, width, conf.angles, device=device)
    return rotator


class LocalFeatureHandler:
    def __call__(
        self,
        path: FilePath,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image
    ) -> LocalFeatureOutputs:
        raise NotImplementedError
    
    def extract_by_keypoints(
        self,
        path: FilePath,
        pre_sampled_keypoints: np.ndarray,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image
    ) -> LocalFeatureOutputs:
        """Get local features based on pre-sampled keypoints
        """
        raise NotImplementedError


class Line2DFeatureHandler:
    def __call__(
        self,
        path: FilePath,
        shape: tuple[int, int],
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image
    ) -> tuple[np.ndarray, Any]:
        raise NotImplementedError


def postprocess(
    outputs: LocalFeatureOutputs,
    img: np.ndarray,
    x: torch.Tensor,
    cropper: Optional[Cropper] = None,
) -> LocalFeatureOutputs:
    lafs, scores, descs = outputs

    # Original image size
    H, W, *_ = img.shape

    # Resized image size
    h, w = x.shape[-2:]

    # Rescale lafs
    lafs[:, 0, :] *= float(W) / float(w)
    lafs[:, 1, :] *= float(H) / float(h)

    if cropper:
        kpts = lafs_to_keypoints(lafs)
        kpts = cropper.convert_cropped_to_original_coordinates_torch(kpts)
        lafs = keypoints_to_lafs(kpts)

    return lafs, scores, descs


def lafs_to_keypoints(lafs: torch.Tensor) -> torch.Tensor:
    """
    Args
    ----
    lafs : Shape(N, 2, 3)
    """
    return kornia.feature.get_laf_center(lafs[None, ...]).reshape(-1, 2)


def keypoints_to_lafs(kpts: torch.Tensor) -> torch.Tensor:
    """
    Args
    ----
    kpts : Shape(N, 2)
    """
    lafs = kornia.feature.laf_from_center_scale_ori(
        kpts[None], torch.ones(1, len(kpts), 1, 1, device=kpts.device)
    )[0]
    return lafs


def filter_keypoints_by_mask(
    kpts: np.ndarray,
    mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args
    ----
    kpts : np.ndarray (shape=(N, 2))
    mask : np.ndarray (shape=(H, W), dtype=bool)
    """
    kpts = kpts.copy()
    kpts = kpts.astype(np.int32)

    H, W = mask.shape

    kpts[:, 0] = np.clip(kpts[:, 0], 0, W - 1)
    kpts[:, 1] = np.clip(kpts[:, 1], 0, H - 1)

    flag = mask[kpts[:, 1], kpts[:, 0]]     # Shape(len(matched_kpts1),)

    keeps, *_ = np.where(flag)

    filtered_kpts = kpts[keeps]
    return filtered_kpts, keeps


def remove_border_keypoints(
    kpts: np.ndarray,
    H: int, W: int,
    pad_size: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    kpts = kpts.copy()
    kpts = kpts.astype(np.int32)

    kpts[:, 0] = np.clip(kpts[:, 0], 0, W - 1)
    kpts[:, 1] = np.clip(kpts[:, 1], 0, H - 1)

    bord = pad_size
    toremoveW = np.logical_or(kpts[:, 0] < bord, kpts[:, 0] >= (W - bord))
    toremoveH = np.logical_or(kpts[:, 1] < bord, kpts[:, 1] >= (H - bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    kpts = kpts[~toremove, :]
    keeps = np.where(~toremove)
    return kpts, keeps
