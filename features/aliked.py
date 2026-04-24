from typing import Callable, Optional

import cv2
import kornia
import torch
from pydantic import BaseModel

from scripts.data import FilePath, resolve_model_path
from features.base import LocalFeatureHandler, LocalFeatureOutputs, create_rotator, postprocess, read_image
from features.config import ALIKEDConfig
from models.aliked.nets.aliked import ALIKED
from scripts.preprocess import resize_image_opencv, resize_image_tensor
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class ALIKEDHandler(LocalFeatureHandler):
    def __init__(self,
                 conf: ALIKEDConfig,
                 device: Optional[torch.device] = None):
        self.conf = conf
        self.device = device
        model = ALIKED(
            model_name=conf.model_name,
            device=device,      # type: ignore
            top_k=conf.top_k,
            scores_th=conf.scores_th,
            n_limit=conf.n_limit,
            load_pretrained=True,
            pretrained_path=str(resolve_model_path(conf.weight_path))
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
            rotator = create_rotator(rotation,
                                     resized_img.shape[0],
                                     resized_img.shape[1],
                                     device=self.device)
            resized_img = rotator.transform_homography_variants(resized_img)[0]

        preds = self.model.run(resized_img)

        kpts = torch.from_numpy(preds['keypoints']).to(self.device, non_blocking=True)  # Shape(N, 2), float(coordinates)
        scores = torch.from_numpy(preds['scores']).to(self.device, non_blocking=True)   # Shape(N,)
        descs = torch.from_numpy(preds['descriptors']).to(self.device, non_blocking=True)   # Shape(N, 128)

        if rotator:
            kpts = rotator.inverse_transform_keypoints_tensor(kpts)

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]    # Shape(1, N, 2, 3) -> Shape(N, 2, 3)

        outputs = (lafs, scores, descs)
        outputs = postprocess(
            outputs, img,
            kornia.utils.image_to_tensor(resized_img, keepdim=False)
        )
        return outputs
