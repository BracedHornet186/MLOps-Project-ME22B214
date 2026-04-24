from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.cuda.amp
from lightglue.utils import numpy_image_to_torch
from PIL import Image

from scripts.data import FilePath, resolve_model_path
from features.base import read_image
from matchers.base import DetectorFreeMatcher
from matchers.config import XFeatStarMatcherConfig
from models.xfeat.xfeat import XFeat
from scripts.preprocess import resize_image_tensor
from preprocesses.config import ResizeConfig
from preprocesses.region import OverlapRegionCropper
from scripts.storage import MatchedKeypointStorage


class XFeatStarMatcher(DetectorFreeMatcher):
    def __init__(
        self, conf: XFeatStarMatcherConfig, device: Optional[torch.device] = None
    ):
        self.conf = conf
        self.device = device or torch.device("cpu")
        self.model = XFeat(weights=str(resolve_model_path(conf.xfeat.weight_path)))

    @torch.inference_mode()
    def __call__(
        self,
        path1: FilePath,
        path2: FilePath,
        matched_keypoint_storage: MatchedKeypointStorage,
        cropper: Optional[OverlapRegionCropper] = None,
        orientation1: Optional[int] = None,
        orientation2: Optional[int] = None,
        image_reader: Callable = read_image,
    ):
        img1 = image_reader(str(path1))
        img2 = image_reader(str(path2))

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        if cropper:
            cropper.set_original_image(img1, img2)
            img1, img2 = cropper.crop_ndarray_image(img1, img2)

        x1 = preprocess(img1, self.device, resize=self.conf.resize)
        x2 = preprocess(img2, self.device, resize=self.conf.resize)

        mkpts1, mkpts2 = self.model.match_xfeat_star(x1, x2, top_k=self.conf.xfeat.topk)

        mkpts1 = postprocess(mkpts1, img1, x1)
        mkpts2 = postprocess(mkpts2, img2, x2)

        if cropper:
            mkpts1, mkpts2 = cropper.convert_cropped_to_original_coordinates(
                mkpts1, mkpts2
            )

        if self.conf.min_matches is None or len(mkpts1) >= self.conf.min_matches:
            # TODO
            scores = np.ones((len(mkpts1),), dtype=np.float32)
            matched_keypoint_storage.add(path1, path2, mkpts1, mkpts2, scores=scores)


def preprocess(
    img: np.ndarray,
    device: torch.device,
    resize: Optional[ResizeConfig] = None,
) -> torch.Tensor:
    x = numpy_image_to_torch(img)
    x = x.to(device, non_blocking=True)
    x = x[None]
    if resize is not None:
        x = resize_image_tensor(x, resize)

    return x


def postprocess(mkpts: np.ndarray, img: np.ndarray, x: torch.Tensor) -> np.ndarray:
    # Original image size
    H, W, *_ = img.shape

    # Resized image size
    h, w = x.shape[-2:]

    # Rescale
    mkpts[:, 0] *= float(W) / float(w)
    mkpts[:, 1] *= float(H) / float(h)

    return mkpts
