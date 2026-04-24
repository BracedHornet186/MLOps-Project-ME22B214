from __future__ import annotations

import tracemalloc
from collections.abc import Callable
from typing import Optional, Tuple, Union

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel

from scripts.data import FilePath, resolve_model_path
from features.base import (
    LocalFeatureHandler,
    LocalFeatureOutputs,
    create_rotator,
    keypoints_to_lafs,
    lafs_to_keypoints,
    postprocess,
    read_image,
)
from features.config import MagicLeapSuperPointConfig
from models._research_only.magicleap.superpoint import SuperPoint
from models._research_only.magicleap.utils import frame2tensor, process_resize
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper
from scripts.workspace import log


class MagicLeapSuperPointHandler(LocalFeatureHandler):
    def __init__(
        self, conf: MagicLeapSuperPointConfig, device: Optional[torch.device] = None
    ):
        weight_path = str(resolve_model_path(conf.weight_path))
        model = SuperPoint(
            {
                "nms_radius": conf.nms_radius,
                "keypoint_threshold": conf.keypoint_threshold,
                "max_keypoints": conf.max_keypoints,
                "remove_borders": conf.remove_borders,
                "model_path": weight_path,
                "fix_sampling": conf.fix_sampling,
            }
        )
        self.conf = conf
        self.device = device
        self.model = model.eval().to(device)

        log(f"Load weights from {weight_path}")
        log("MagicLeapSuperPoint has a research-only license !!")

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
        image = image_reader(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if cropper:
            cropper.set_original_image(image)
            image = cropper.crop_ndarray_image_gray(image)

        if resize:
            assert resize.func == "magicleap"
            # NOTE:
            # Since the image is grayscale, the shape is (H, W)
            W, H = image.shape[1], image.shape[0]
            W_new, H_new = process_resize(W, H, [resize.ml_resize])
            scales = np.array(
                [float(W) / float(W_new), float(H) / float(H_new)], dtype=np.float32
            )
            resized_image = cv2.resize(image, (W_new, H_new)).astype(np.float32)
        else:
            raise NotImplementedError

        x = frame2tensor(resized_image, self.device)

        rotator = None
        if rotation:
            rotator = create_rotator(rotation, x.shape[-2], x.shape[-1], self.device)
            x = rotator.transform_homography_variants_tensor(x)
            assert x.shape[0] == 1

        data = {"image": x}
        preds = self.model(data)

        kpts = preds["keypoints"][0]  # Shape(N, 2)
        scores = preds["scores"][0]  # Shape(N,)
        descs = preds["descriptors"][0]  # Shape(dim, N)
        descs = descs.T

        if rotator:
            kpts = rotator.inverse_transform_keypoints_tensor(kpts)

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]  # Shape(1, N, 2, 3) -> Shape(N, 2, 3)

        # Converting keypoints to lafs
        # Set scale to 1
        # lafs = torch.cat([
        #    torch.eye(2, device=kpts.device, dtype=kpts.dtype).unsqueeze(0).unsqueeze(1).expand(
        #        kpts.size(0), kpts.size(1), -1, -1
        #    ),
        #    kpts.unsqueeze(-1),
        # ], dim=-1)

        # Rescale lafs
        lafs[:, 0, :] *= scales[0]
        lafs[:, 1, :] *= scales[1]

        if cropper:
            kpts = lafs_to_keypoints(lafs)
            kpts = cropper.convert_cropped_to_original_coordinates_torch(kpts)
            lafs = keypoints_to_lafs(kpts)

        outputs = (lafs.float(), scores.float(), descs.float())
        return outputs
