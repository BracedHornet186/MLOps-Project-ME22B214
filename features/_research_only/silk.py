from copy import deepcopy
from typing import Callable, Optional, Tuple, Union

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
    keypoints_to_lafs,
    lafs_to_keypoints,
    read_image,
)
from features.config import SiLKConfig
from models._research_only.silk.backbones.silk.silk import SiLKVGG as SiLK
from models._research_only.silk.backbones.silk.silk import (
    from_feature_coords_to_image_coords,
)
from models._research_only.silk.backbones.superpoint.vgg import ParametricVGG
from models._research_only.silk.config.model import load_model_from_checkpoint
from postprocesses.nms import nms_local_features
from scripts.preprocess import resize_image_opencv
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper
from scripts.workspace import log

SILK_DEFAULT_OUTPUT = (  # outputs required when running the model
    "dense_positions",
    "normalized_descriptors",
    "probability",
)
SILK_SCALE_FACTOR = 1.41  # scaling of descriptor output, do not change


class SiLKHandler(LocalFeatureHandler):
    def __init__(self,
                 conf: SiLKConfig,
                 device: Optional[torch.device] = None):
        weight_path = str(resolve_model_path(conf.weight_path))
        device = device or torch.device('cpu')

        # load model
        backbone = ParametricVGG(
            use_max_pooling=False,
            padding=0,
            normalization_fn=[torch.nn.BatchNorm2d(i) for i in (64, 64, 128, 128)],
        )
        model = SiLK(
            in_channels=1,
            backbone=backbone,
            detection_threshold=conf.threshold,
            detection_top_k=conf.topk,
            nms_dist=0,
            border_dist=conf.border_dist,
            default_outputs=SILK_DEFAULT_OUTPUT,
            descriptor_scale_factor=SILK_SCALE_FACTOR,
            padding=0,
        )
        model = load_model_from_checkpoint(
            model,
            checkpoint_path=weight_path,
            state_dict_fn=lambda x: {k[len("_mods.model.") :]: v for k, v in x.items()},
            device=device,
            freeze=True,
            eval=True,
        )
        self.conf = conf
        self.device = device
        self.model = model.eval().to(device)

        log(f'Load weights from {weight_path}')
        log(f'SiLK has a research-only license !!')

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
        img = image_reader(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if cropper:
            cropper.set_original_image(img)
            img = cropper.crop_ndarray_image_gray(img)

        if resize:
            resized_img, scale, mask = resize_image_opencv(img, conf=resize)
        else:
            resized_img = img
            scale = None

        x = torch.tensor(resized_img,
                         device=self.device,
                         dtype=torch.float32)
        x = x.unsqueeze(0).unsqueeze(0) / 255.0

        kpts, descs, probs = self.model(x)
        kpts = from_feature_coords_to_image_coords(self.model, kpts)
        descs = descs.reshape(1, 128, -1).permute(0, 2, 1)

        kpts = kpts[0]      # Shape(N, 3)
        probs = probs[0]   # Shape(1, H', W')
        descs = descs[0]    # Shape(dim, H', W')

        if scale is not None:
            kpts[:, 0] *= scale[0]
            kpts[:, 1] *= scale[1]
        
        kpts, descs, scores = get_top_k(kpts, descs, self.conf.topk)
        kpts = kpts[:, [1,0]]  # yx -> xy
        descs = descs * 1.41

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]    # Shape(1, N, 2, 3) -> Shape(N, 2, 3)

        if cropper:
            kpts = lafs_to_keypoints(lafs)
            kpts = cropper.convert_cropped_to_original_coordinates_torch(kpts)
            lafs = keypoints_to_lafs(kpts)

        if self.conf.nms:
            lafs, scores, descs = nms_local_features(
                lafs, scores, descs, img, self.conf.nms
            )

        outputs = (lafs, scores, descs)
        return outputs
    

def get_top_k(keypoints: torch.Tensor,
              descriptors: torch.Tensor,
              k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions, scores = keypoints[:,:2], keypoints[:,2]

    # top-k selection
    idxs = scores.argsort()[-k:]

    return positions[idxs], descriptors[idxs] / 1.41, scores[idxs]