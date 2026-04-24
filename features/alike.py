from typing import Optional, Callable

import cv2
import kornia
import torch
from pydantic import BaseModel

from scripts.data import FilePath, resolve_model_path
from features.base import LocalFeatureHandler, LocalFeatureOutputs, postprocess, read_image
from features.config import ALIKEConfig
from models.alike.alike import ALike, configs
from scripts.preprocess import resize_image_opencv, resize_image_tensor
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class ALIKEHandler(LocalFeatureHandler):
    def __init__(self,
                 conf: ALIKEConfig,
                 device: Optional[torch.device] = None):
        self.conf = conf
        self.device = device

        _config = configs[conf.model_type].copy()
        _config['model_path'] = resolve_model_path(conf.weight_path)

        model = ALike(**_config,
                      device=device,        # type: ignore
                      top_k=conf.top_k,
                      scores_th=conf.scores_th,
                      n_limit=conf.n_limit)
        self.model = model

    def __call__(
        self,
        path: FilePath,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image
    ) -> LocalFeatureOutputs:
        img = image_reader(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize:
            resized_img, scale, mask = resize_image_opencv(img, conf=resize)
        else:
            resized_img = img

        rotator = None
        if rotation:
            raise NotImplementedError

        preds = self.model(resized_img, sub_pixel=self.conf.sub_pixel)

        kpts = torch.from_numpy(preds['keypoints']).to(self.device, non_blocking=True)
        scores = torch.from_numpy(preds['scores']).to(self.device, non_blocking=True)
        descs = torch.from_numpy(preds['descriptors']).to(self.device, non_blocking=True)

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]    # Shape(1, N, 2, 3) -> Shape(N, 2, 3)

        outputs = (lafs, scores, descs)
        outputs = postprocess(
            outputs, img,
            kornia.utils.image_to_tensor(resized_img, keepdim=False)
        )
        return outputs
