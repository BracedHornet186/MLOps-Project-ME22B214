from typing import Callable, Optional

import kornia
import torch
from lightglue.disk import DISK, Extractor
from lightglue.utils import numpy_image_to_torch

from data import FilePath, resolve_model_path
from features.base import (
    LocalFeatureHandler,
    LocalFeatureOutputs,
    keypoints_to_lafs,
    lafs_to_keypoints,
    read_image,
)
from features.config import LightGlueDISKConfig
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class _DISK(DISK):
    def __init__(self, weight_path: str, **conf) -> None:
        Extractor.__init__(self, **conf)  # Update with default configuration.
        model = kornia.feature.DISK()
        weight = torch.load(resolve_model_path(weight_path), map_location="cpu")
        model.load_state_dict(weight["extractor"])
        self.model = model.eval()
        print(f"[DISK] Loaded weight from {resolve_model_path(weight_path)}")


class LightGlueDISKHandler(LocalFeatureHandler):
    def __init__(
        self, conf: LightGlueDISKConfig, device: Optional[torch.device] = None
    ):
        self.conf = conf
        self.device = device
        model = (
            _DISK(conf.weight_path, max_num_keypoints=conf.max_num_keypoints)
            .eval()
            .to(device)
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

        if cropper:
            cropper.set_original_image(img)
            img = cropper.crop_tensor_image(img)

        rotator = None
        if rotation:
            raise NotImplementedError

        if resize:
            raise NotImplementedError
        else:
            preds = self.model.extract(img)

        kpts = preds["keypoints"].reshape(-1, 2)
        scores = preds["keypoint_scores"].reshape(-1)
        descs = preds["descriptors"].reshape(len(kpts), -1)

        if rotator:
            raise NotImplementedError

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]  # Shape(1, N, 2, 3) -> Shape(N, 2, 3)

        if cropper:
            kpts = lafs_to_keypoints(lafs)
            kpts = cropper.convert_cropped_to_original_coordinates_torch(kpts)
            lafs = keypoints_to_lafs(kpts)

        outputs = (lafs, scores, descs)
        return outputs
