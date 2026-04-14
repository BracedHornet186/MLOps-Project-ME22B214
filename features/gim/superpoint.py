from typing import Callable, Optional

import cv2
import numpy as np
import kornia
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from kornia.color import grayscale_to_rgb
from lightglue.utils import numpy_image_to_torch

from data import FilePath, resolve_model_path
from features.base import (
    LocalFeatureHandler,
    LocalFeatureOutputs,
    keypoints_to_lafs,
    lafs_to_keypoints,
    read_image,
    create_rotator,
    postprocess,
)
from models.gim.gluefactory.superpoint import SuperPoint
from features.config import GIMSuperPointConfig
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper
from preprocesses.orientation import OrientationNormalizer


class GIMSuperPointHandler(LocalFeatureHandler):
    def __init__(
        self, conf: GIMSuperPointConfig, device: Optional[torch.device] = None
    ):
        self.conf = conf
        self.device = device
        model = SuperPoint(
            {
                "max_num_keypoints": conf.max_num_keypoints,
                "force_num_keypoints": True,
                "detection_threshold": 0.0,
                "nms_radius": 3,
                "trainable": False,
            }
        )
        checkpoints_path = str(resolve_model_path(conf.weight_path))
        state_dict = torch.load(checkpoints_path, map_location="cpu")
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith("model."):
                state_dict.pop(k)
            if k.startswith("superpoint."):
                state_dict[k.replace("superpoint.", "", 1)] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        self.model = model.eval().to(device)

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
        assert resize
        assert resize.gim_resize

        ori_normalizer = None
        if orientation is not None:
            ori_normalizer = OrientationNormalizer(degree=orientation)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if cropper:
            cropper.set_original_image(img)
            img = cropper.crop_ndarray_image_gray(img)

        if ori_normalizer:
            raise NotImplementedError

        rotator = None
        if rotation:
            raise NotImplementedError

        img, scale = preprocess(img, grayscale=True, resize_max=resize.gim_resize)
        img = img.to(self.device, non_blocking=True)[None]

        data = {'image': img}
        preds = self.model(data)

        kpts = preds["keypoints"].reshape(-1, 2)
        scores = preds["scores"].reshape(-1)
        descs = preds["descriptors"].reshape(len(kpts), -1)

        if rotator:
            raise NotImplementedError

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]  # Shape(1, N, 2, 3) -> Shape(N, 2, 3)

        # Rescale lafs
        lafs[:, 0, :] *= scale[0]
        lafs[:, 1, :] *= scale[1]

        if ori_normalizer:
            raise NotImplementedError

        if cropper:
            kpts = lafs_to_keypoints(lafs)
            kpts = cropper.convert_cropped_to_original_coordinates_torch(kpts)
            lafs = keypoints_to_lafs(kpts)

        outputs = (lafs, scores, descs)
        return outputs

    @torch.inference_mode()
    def extract_by_keypoints(
        self,
        path: FilePath,
        pre_sampled_keypoints: np.ndarray,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image,
    ) -> LocalFeatureOutputs:
        raise NotImplementedError


def preprocess(
    image: np.ndarray,
    grayscale: bool = False,
    resize_max: Optional[int] = None,
    dfactor: int = 8,
) -> tuple[torch.Tensor, np.ndarray]:
    image = image.astype(np.float32, copy=False)
    size = image.shape[:2][::-1]
    scale = np.array([1.0, 1.0])

    if resize_max:
        scale = resize_max / max(size)
        if scale < 1.0:
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, "cv2_area")
            scale = np.array(size) / np.array(size_new)

    if grayscale:
        assert image.ndim == 2, image.shape
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = torch.from_numpy(image / 255.0).float()

    # assure that the size is divisible by dfactor
    size_new = tuple(map(lambda x: int(x // dfactor * dfactor), image.shape[-2:]))
    image = F.resize(image, size=size_new)
    scale = np.array(size) / np.array(size_new)[::-1]
    return image, scale


def resize_image(image, size, interp):
    assert interp.startswith('cv2_')
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    # elif interp.startswith('pil_'):
    #     interp = getattr(PIL.Image, interp[len('pil_'):].upper())
    #     resized = PIL.Image.fromarray(image.astype(np.uint8))
    #     resized = resized.resize(size, resample=interp)
    #     resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized