from typing import Callable, Optional, Tuple, Union

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel

from scripts.data import FilePath, resolve_model_path
from features.base import (LocalFeatureHandler, LocalFeatureOutputs,
                           create_rotator, keypoints_to_lafs,
                           lafs_to_keypoints, postprocess, read_image)
from features.config import SuperPointConfig
from models.superpoint.model import SuperPointNetBn
from scripts.preprocess import resize_image_opencv, resize_image_tensor
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class SuperPointHandler(LocalFeatureHandler):
    def __init__(self,
                 conf: SuperPointConfig,
                 device: Optional[torch.device] = None):
        weight_path = str(resolve_model_path(conf.weight_path))
        model = SuperPointNetBn(
            max_keypoints=conf.max_keypoints,
            nms_kernel=conf.nms_kernel,
            remove_borders_size=conf.remove_borders_size,
            keypoint_threshold=conf.keypoint_threshold,
            weights=weight_path
        )
        self.conf = conf
        self.device = device
        self.model = model.eval().cuda()

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
        image = image_reader(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        target_size = None
        if resize and resize.target_size_height and resize.target_size_width:
            target_size = (
                resize.target_size_width,   # X
                resize.target_size_height   # Y
            )
        resized_image = preprocess(image, target_size=target_size)
        x = (torch.FloatTensor(resized_image) / 255.).unsqueeze(0).unsqueeze(0).to(self.device, non_blocking=True)

        rotator = None
        if rotation:
            rotator = create_rotator(rotation, x.shape[-2], x.shape[-1], self.device)
            x = rotator.transform_homography_variants_tensor(x)
            assert x.shape[0] == 1

        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = self.model(x)
        lafs = preds[0][0]
        scores = preds[1][0]
        descs = preds[2][0]

        if rotator:
            kpts = lafs_to_keypoints(lafs)
            kpts = rotator.inverse_transform_keypoints_tensor(kpts)
            lafs = keypoints_to_lafs(kpts)

        # Original image size
        H, W, *_ = image.shape

        # Resized image size
        h, w = x.shape[-2:]

        # Rescale lafs
        lafs[:, 0, :] *= float(W) / float(w)
        lafs[:, 1, :] *= float(H) / float(h)

        outputs = (lafs, scores, descs)
        return outputs
    

def preprocess(image: np.ndarray,
               target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Adapted from https://github.com/ucuapps/OpenGlue/blob/main/extract_features.py

    Read image, convert to gray and resize to target size
    Args:
        image: image object
        target_size: size of returned image will correspond with target size in at least one dimension, such that
        its aspect ratio is preserved

    Returns:
        image: array (H, W) representing image, where H=target_size[1] or W=target_size[0]
    """
    # read image and convert to gray
    size = image.shape[:2][::-1]

    if target_size is not None:
        # resize image to target size
        target_ratio = target_size[0] / target_size[1]  # 960 / 720 = 1.333
        current_ratio = size[0] / size[1]
        if current_ratio > target_ratio:
            resize_height = target_size[1]
            resize_width = int(current_ratio * resize_height)
            image = cv2.resize(image, (resize_width, resize_height))
        else:
            resize_width = target_size[0]
            resize_height = int(resize_width / current_ratio)
            image = cv2.resize(image, (resize_width, resize_height))
    return image
