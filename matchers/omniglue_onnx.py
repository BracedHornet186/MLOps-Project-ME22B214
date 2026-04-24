from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.cuda.amp

from scripts.data import FilePath, resolve_model_path
from features.base import read_image
from matchers.base import DetectorFreeMatcher
from matchers.config import OmniGlueONNXConfig
try:
    from models.omniglue_onnx import OmniGlue
except Exception:
    print("Notfound: onnxruntime")
    OmniGlue = None
from scripts.preprocess import resize_image_opencv
from preprocesses.config import ResizeConfig
from preprocesses.region import OverlapRegionCropper
from scripts.storage import MatchedKeypointStorage


class OmniGlueONNXMatcher(DetectorFreeMatcher):
    def __init__(
        self, conf: OmniGlueONNXConfig, device: Optional[torch.device] = None
    ):
        self.conf = conf
        self.device = device or torch.device("cpu")
        self.model = OmniGlue(
            og_export=str(resolve_model_path(conf.weight_path)),
            sp_export=str(resolve_model_path(conf.weight_path_superpoint)),
            dino_export=str(resolve_model_path(conf.weight_path_dinov2))
        )

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

        x1 = preprocess(img1, resize=self.conf.resize)
        x2 = preprocess(img2, resize=self.conf.resize)

        mkpts1, mkpts2, scores = self.model.FindMatches(x1, x2, max_keypoints=self.conf.max_keypoints)

        mkpts1 = postprocess(mkpts1, img1, x1)
        mkpts2 = postprocess(mkpts2, img2, x2)

        if cropper:
            mkpts1, mkpts2 = cropper.convert_cropped_to_original_coordinates(
                mkpts1, mkpts2
            )

        if self.conf.min_matches is None or len(mkpts1) >= self.conf.min_matches:
            matched_keypoint_storage.add(path1, path2, mkpts1, mkpts2, scores=scores)


def preprocess(
    img: np.ndarray,
    resize: Optional[ResizeConfig] = None,
) -> np.ndarray:
    if resize is not None:
        assert not resize.pad_bottom_right
        img, _, _ = resize_image_opencv(img, resize, order3ch='hwc')
    return img


def postprocess(mkpts: np.ndarray, img: np.ndarray, x: np.ndarray) -> np.ndarray:
    # Original image size
    H, W, *_ = img.shape

    # Resized image size
    h, w, *_ = x.shape

    # Rescale
    mkpts[:, 0] *= float(W) / float(w)
    mkpts[:, 1] *= float(H) / float(h)

    return mkpts
