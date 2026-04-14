from typing import Callable, Optional

import cv2
import kornia
import numpy as np
import torch

from data import FilePath, resolve_model_path
from matchers.base import DetectorFreeMatcher
from matchers.config import LoFTRConfig
from preprocess import paired_pre_resize, resize_image_tensor
from preprocesses.config import ResizeConfig
from preprocesses.orientation import OrientationNormalizer
from preprocesses.region import OverlapRegionCropper
from storage import MatchedKeypointStorage
from workspace import log, perf_time


def read_image(path: FilePath) -> np.ndarray:
    return cv2.imread(str(path))


class LoFTRMatcher(DetectorFreeMatcher):
    def __init__(self, conf: LoFTRConfig, device: Optional[torch.device] = None):
        model = kornia.feature.LoFTR(pretrained=None)
        weight = torch.load(resolve_model_path(conf.weight_path), map_location="cpu")[
            "state_dict"
        ]
        model.load_state_dict(weight)
        model = model.eval().to(device)
        log(f"[LoFTRMatcher] model loaded weights from {conf.weight_path}")
        log(f"[LoFTRMatcher] Use the device ({device})")
        self.model = model
        self.conf = conf
        self.device = device

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
        with perf_time("read_image"):
            img1, img2 = image_reader(str(path1)), image_reader(str(path2))

        ori_normalizer1 = OrientationNormalizer.create_if_needed(orientation1)
        ori_normalizer2 = OrientationNormalizer.create_if_needed(orientation2)

        with perf_time("crop_overlap_regions"):
            if cropper:
                cropper.set_original_image(img1, img2)
                img1, img2 = cropper.crop_ndarray_image(img1, img2)

        with perf_time("normalize_orientation"):
            if ori_normalizer1:
                ori_normalizer1.set_original_image(img1)
                img1 = ori_normalizer1.get_upright_image_ndarray()
            if ori_normalizer2:
                ori_normalizer2.set_original_image(img2)
                img2 = ori_normalizer2.get_upright_image_ndarray()

        with perf_time("paired_pre_resize"):
            if self.conf.paired_pre_resize:
                _img1, _img2 = paired_pre_resize(
                    img1, img2, conf=self.conf.paired_pre_resize
                )
            else:
                _img1, _img2 = img1, img2

        with perf_time("preprocess"):
            x1 = preprocess(_img1, resize=self.conf.resize, device=self.device)
            x2 = preprocess(_img2, resize=self.conf.resize, device=self.device)

        with perf_time("model"):
            outputs = self.model({"image0": x1, "image1": x2})

        mkpts1: np.ndarray = outputs["keypoints0"].cpu().numpy()
        mkpts2: np.ndarray = outputs["keypoints1"].cpu().numpy()
        scores = outputs['confidence'].cpu().numpy()
        idx = np.argsort(-scores)

        mkpts1 = mkpts1[idx]
        mkpts2 = mkpts2[idx]
        scores = scores[idx]

        if self.conf.confidence_threshold is not None:
            keep = scores >= self.conf.confidence_threshold
            mkpts1 = mkpts1[keep]
            mkpts2 = mkpts2[keep]
            scores = scores[keep]
        
        if self.conf.topk is not None:
            k = self.conf.topk
            mkpts1 = mkpts1[:k]
            mkpts2 = mkpts2[:k]
            scores = scores[:k]
        
        mkpts1 = postprocess(mkpts1, img1, x1)
        mkpts2 = postprocess(mkpts2, img2, x2)

        if ori_normalizer1:
            mkpts1 = ori_normalizer1.keypoints_to_original_coords_ndarray(mkpts1)
        if ori_normalizer2:
            mkpts2 = ori_normalizer2.keypoints_to_original_coords_ndarray(mkpts2)

        if cropper:
            mkpts1, mkpts2 = cropper.convert_cropped_to_original_coordinates(
                mkpts1, mkpts2
            )

        if self.conf.min_matches is None or len(mkpts1) >= self.conf.min_matches:
            matched_keypoint_storage.add(path1, path2, mkpts1, mkpts2, scores=scores)


def preprocess(
    img: np.ndarray,
    resize: Optional[ResizeConfig] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    x = kornia.utils.image_to_tensor(img, keepdim=False).float() / 255.0
    if device:
        x = x.to(device, non_blocking=True)
    x = kornia.color.bgr_to_rgb(x)
    x = kornia.color.rgb_to_grayscale(x)
    if resize:
        x = resize_image_tensor(x, resize)
    return x


def postprocess(kpts: np.ndarray, img: np.ndarray, x: torch.Tensor) -> np.ndarray:
    H, W, *_ = img.shape
    h, w = x.shape[-2:]

    kpts = kpts.copy()
    kpts[:, 0] *= float(W) / float(w)
    kpts[:, 1] *= float(H) / float(h)

    return kpts
