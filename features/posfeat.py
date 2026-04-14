from typing import Optional, Callable

import cv2
import kornia
import numpy as np
import torch

from data import FilePath, resolve_model_path
from features.base import (LocalFeatureHandler, LocalFeatureOutputs, create_rotator, read_image,
                           remove_border_keypoints)
from features.config import PosFeatConfig
from models.posfeat.extractor import PosFeatExtractor
from preprocess import resize_image_opencv
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class PosFeatHandler(LocalFeatureHandler):
    def __init__(self,
                 conf: PosFeatConfig,
                 device: Optional[torch.device] = None):
        self.conf = conf
        self.device = device
        weight_path = str(resolve_model_path(conf.weight_path))
        self.model = PosFeatExtractor(
            weight_path,
            model_name=conf.model,
            detector_name=conf.detector,
            detector_conf=conf.detector_config,
            loss_distance=conf.loss_distance,
            device=device
        )

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
        """Override
        """
        assert resize
        img = image_reader(str(path))
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        resized_img, scale, mask = resize_image_opencv(img, conf=resize)
        x = self.to_torch_image(resized_img)
        outputs = self.extract(x, scale=scale, orig_img=img)
        return outputs

    def to_torch_image(self, img: np.ndarray) -> torch.Tensor:
        return self.model.transform(img).unsqueeze(0)    # type: ignore
    
    def extract(self,
                x: torch.Tensor,
                scale: np.ndarray,
                orig_img: np.ndarray,
                rotation: Optional[RotationConfig] = None,
                *args, **kwargs) -> LocalFeatureOutputs:
        rotator = None
        if rotation:
            raise NotImplementedError

        preds = self.model.extract({'im1': x, 'scale': scale})

        # To numpy once
        preds['desc'] = preds['desc'].cpu().numpy()[0]
        preds['kp_score'] = preds['kp_score'].cpu().numpy()[0]

        if self.conf.remove_border_pad_size > 0:
            _kpts: np.ndarray = preds['kpt']     # type: ignore
            H, W, *_ = orig_img.shape

            _, keeps = remove_border_keypoints(
                _kpts, H, W,
                pad_size=self.conf.remove_border_pad_size
            )
            preds['kpt'] = preds['kpt'][keeps]
            preds['desc'] = preds['desc'][keeps]
            preds['kp_score'] = preds['kp_score'][keeps]

        scores: torch.Tensor = torch.from_numpy(preds['kp_score']).to(self.device, non_blocking=True)
        descs: torch.Tensor = torch.from_numpy(preds['desc']).to(self.device, non_blocking=True)      # Shape(1, N, dim) -> (N, dim)
        kpts: torch.Tensor = torch.from_numpy(preds['kpt']).to(self.device, non_blocking=True)

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]
        return lafs, scores, descs