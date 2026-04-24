from typing import Callable, Optional

import kornia
import torch
import torch.nn as nn
from lightglue.superpoint import Extractor, SuperPoint
from lightglue.utils import numpy_image_to_torch

from scripts.data import FilePath, resolve_model_path
from features.base import (
    LocalFeatureHandler,
    LocalFeatureOutputs,
    create_rotator,
    keypoints_to_lafs,
    lafs_to_keypoints,
    read_image,
)
from features.config import LightGlueSuperPointConfig
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class _SuperPoint(SuperPoint):
    def __init__(self, weight_path: str, **conf):
        Extractor.__init__(self, **conf)  # Update with default configuration.
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.conf.descriptor_dim, kernel_size=1, stride=1, padding=0
        )

        weight = torch.load(weight_path, map_location='cpu')
        self.load_state_dict(weight)
        print(f"[LightGlueSuperPoint] weight={weight_path}")

        if self.conf.max_num_keypoints is not None and self.conf.max_num_keypoints <= 0:
            raise ValueError("max_num_keypoints must be positive or None")


class LightGlueSuperPointHandler(LocalFeatureHandler):
    def __init__(
        self, conf: LightGlueSuperPointConfig, device: Optional[torch.device] = None
    ):
        self.conf = conf
        self.device = device
        model = (
            _SuperPoint(
                weight_path=str(resolve_model_path(conf.weight_path)),
                max_num_keypoints=conf.max_num_keypoints,
                detection_threshold=conf.detection_threshold,
                nms_radius=conf.nms_radius,
                remove_borders=conf.remove_borders
            )
            .eval()
            .to(self.device, dtype=torch.float32)
        )
        self.model = model

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
        img = image_reader(str(path))
        img = img[..., ::-1]  # BGR->RGB
        img = numpy_image_to_torch(img)
        img = img.to(self.device, non_blocking=True)
        img = img[None]

        if cropper:
            cropper.set_original_image(img[0])
            img = cropper.crop_tensor_image(img[0])
            img = img[None, ...]  # Shape(1, 3, H', W')

        rotator = None
        if resize:
            resize_conf = {
                "resize": resize.lg_resize,  # target edge length, None for no resizing
                "side": "long",
                "interpolation": "bilinear",
                "align_corners": None,
                "antialias": True,
            }
            x = kornia.geometry.transform.resize(
                img,
                resize_conf["resize"],
                side=resize_conf["side"],
                antialias=resize_conf["antialias"],
                align_corners=resize_conf["align_corners"],
            )
        else:
            x = img

        if rotation:
            rotator = create_rotator(
                rotation, x.shape[-2], x.shape[-1], device=self.device
            )
            x = rotator.transform_homography_variants_tensor(x)

        preds = self.model.extract(x, resize=None)

        kpts = preds["keypoints"].reshape(-1, 2)
        scores = preds["keypoint_scores"].reshape(-1)
        descs = preds["descriptors"].reshape(len(kpts), -1)

        if rotator:
            kpts = rotator.inverse_transform_keypoints_tensor(kpts)

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]  # Shape(1, N, 2, 3) -> Shape(N, 2, 3)

        # Original image size
        H, W = img.shape[-2:]

        # Resized image size
        h, w = x.shape[-2:]

        # Rescale lafs
        lafs[:, 0, :] *= float(W) / float(w)
        lafs[:, 1, :] *= float(H) / float(h)

        if cropper:
            kpts = lafs_to_keypoints(lafs)
            kpts = cropper.convert_cropped_to_original_coordinates_torch(kpts)
            lafs = keypoints_to_lafs(kpts)

        outputs = (lafs, scores, descs)
        return outputs
