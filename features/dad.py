from __future__ import annotations

from collections.abc import Callable

import cv2
import dad
import kornia.feature
import numpy as np
import torch
from dad.utils import check_not_i16, sample_keypoints
from PIL import Image

from data import FilePath, resolve_model_path
from features.base import (
    LocalFeatureHandler,
    LocalFeatureOutputs,
    keypoints_to_lafs,
    lafs_to_keypoints,
    postprocess,
    read_image,
)
from features.config import DaDConfig
from preprocess import resize_image_tensor
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class DaDHandler(LocalFeatureHandler):
    def __init__(self, conf: DaDConfig, device: torch.device | None = None):
        self.conf = conf
        self.device = device
        model = dad.load_DaD(pretrained=False)
        weights = torch.load(resolve_model_path(conf.weight_path), weights_only=False)
        model.load_state_dict(weights)
        self.model = model.eval().to(device)

    def __call__(
        self,
        path: FilePath,
        resize: ResizeConfig | None = None,
        rotation: RotationConfig | None = None,
        cropper: Cropper | None = None,
        orientation: int | None = None,
        image_reader: Callable = read_image,
    ) -> LocalFeatureOutputs:
        if rotation:
            raise NotImplementedError
        img = image_reader(str(path))
        H, W, _ = img.shape

        if cropper:
            cropper.set_original_image(img)
            img = cropper.crop_ndarray_image(img)

        if orientation is not None:
            raise NotImplementedError

        if rotation:
            raise NotImplementedError

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        batch = self._load_image(img)

        detections = self.model.detect(
            batch, num_keypoints=self.conf.num_keypoints, return_dense_probs=True
        )
        kpts = self.model.to_pixel_coords(
            detections["keypoints"], H, W
        )  # Shape(1, #kpts, 2)
        scores = detections["keypoint_probs"]  # Shape(1, 512)

        kpts = kpts[0]  # Shape(#kpts, 2)
        scores = scores[0]  # Shape(#kpts,)

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]  # Shape(1, N, 2, 3) -> Shape(N, 2, 3)

        if cropper:
            kpts = lafs_to_keypoints(lafs)
            kpts = cropper.convert_cropped_to_original_coordinates_torch(kpts)
            lafs = keypoints_to_lafs(kpts)

        # NOTE:
        # DaD extracts only keypoints, so generates dummy descriptors
        descs = torch.zeros((len(kpts), 256), dtype=torch.float32).to(kpts.device)
        outputs = (lafs, scores, descs)
        return outputs

    def _load_image(self, img: np.ndarray) -> dict[str, torch.Tensor]:
        """Adapted from dad.detectors.dedode_detector"""
        pil_im = Image.fromarray(img)
        check_not_i16(pil_im)
        pil_im = pil_im.convert("RGB")
        if self.model.keep_aspect_ratio:
            W, H = pil_im.size
            scale = self.model.resize / max(W, H)
            W = int((scale * W) // 8 * 8)
            H = int((scale * H) // 8 * 8)
        else:
            H, W = self.model.resize, self.model.resize
        pil_im = pil_im.resize((W, H))
        standard_im = np.array(pil_im) / 255.0
        return {
            "image": self.model.normalizer(
                torch.from_numpy(standard_im).permute(2, 0, 1)
            )
            .float()
            .to(self.device)[None]
        }
