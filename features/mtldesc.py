from typing import Callable, Optional

import cv2
import kornia
import numpy as np
import torch
from kornia.utils.image import image_to_tensor

from data import FilePath, resolve_model_path
from features.base import (LocalFeatureHandler, LocalFeatureOutputs, create_rotator,
                           postprocess, read_image)
from features.config import MTLDescConfig
from models.mtldesc.mtldesc import MTLDescExtractor
from preprocess import resize_image_tensor
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class MTLDescHandler(LocalFeatureHandler):
    def __init__(self,
                 conf: MTLDescConfig,
                 device: Optional[torch.device] = None):
        self.conf = conf
        self.device = device
        conf.model.weight_path = str(resolve_model_path(conf.model.weight_path))
        self.model = MTLDescExtractor(device=device, **conf.model.dict())

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
        img = image_reader(path)
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        
        if cropper:
            cropper.set_original_image(img)
            img = cropper.crop_ndarray_image(img)

        if resize is not None:
            resized_img_tensor = image_to_tensor(img, False).float() / 255.
            resized_img_tensor = (resize_image_tensor(resized_img_tensor, resize) * 255).int()
            resized_img = resized_img_tensor.numpy().astype(np.uint8)[0].transpose((1, 2, 0))
        else:
            resized_img_tensor = image_to_tensor(img, False).float() / 255.
            resized_img = img

        rotator = None
        if rotation:
            rotator = create_rotator(rotation,
                                     resized_img.shape[0],
                                     resized_img.shape[1],
                                     device=self.device)
            resized_img = rotator.transform_homography_variants(resized_img)[0]

        res = self.model.predict(img=resized_img)

        if False:
            X, Y, S, C, Q, D = [], [], [], [], [], []
            x = res['keypoints'][:, 0]
            y = res['keypoints'][:, 1]
            d = res['descriptors']
            scores = res['scores']

            X.append(x)
            Y.append(y)
            C.append(scores)
            D.append(d)
            Y = np.hstack(Y)
            X = np.hstack(X)
            scores = np.hstack(C)
            XY = np.stack([X, Y])
            XY = np.swapaxes(XY, 0, 1)
            D = np.vstack(D)
            idxs = scores.argsort()[-self.conf.topk or None:]
        else:
            XY = res['keypoints']
            D = res['descriptors']
            scores = res['scores']
            idxs = (-scores).argsort()[:self.conf.topk]

        kpts = torch.from_numpy(XY[idxs]).to(self.device, non_blocking=True)
        scores = torch.from_numpy(scores[idxs]).to(self.device, non_blocking=True)
        descs = torch.from_numpy(D[idxs]).to(self.device, non_blocking=True)

        if rotator:
            kpts = rotator.inverse_transform_keypoints_tensor(kpts)

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]
        outputs = (lafs, scores, descs)
        outputs = postprocess(outputs, img, resized_img_tensor, cropper=cropper)
        return outputs
