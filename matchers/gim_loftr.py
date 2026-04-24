from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import cv2
import numpy as np
import torch
import torch.cuda.amp
from PIL import Image

from scripts.data import FilePath, resolve_model_path
from features.gim.superpoint import preprocess, resize_image
from matchers.base import DetectorFreeMatcher
from matchers.config import GIMLoFTRConfig
from models.gim.loftr.config import get_cfg_defaults
from models.gim.loftr.loftr import LoFTR
from models.gim.loftr.misc import lower_config
from pipelines.verification import run_ransac
from postprocesses.nms import nms_matched_keypoints, sort_matched_keypoints_by_score
from postprocesses.panet import PANetRefiner
from preprocesses.orientation import OrientationNormalizer
from preprocesses.region import OverlapRegionCropper
from scripts.storage import MatchedKeypointStorage
from scripts.workspace import log


def read_image(path: str) -> np.ndarray:
    return cv2.imread(str(path))


class GIMLoFTRMatcher(DetectorFreeMatcher):
    def __init__(self, conf: GIMLoFTRConfig, device: Optional[torch.device] = None):
        weight_path = resolve_model_path(conf.weight_path)
        model = LoFTR(lower_config(get_cfg_defaults())["loftr"])

        state_dict = torch.load(weight_path, map_location="cpu")
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)

        log(f"[GIMLoFTRMatcher] Weights were loaded from {weight_path}")
        log(f"[GIMLoFTRMatcher] Use the device ({device})")
        self.model = model.eval().to(device)
        self.conf = conf
        self.device = device or torch.device("cuda")
        assert self.conf.resize
        assert self.conf.resize.func == "gim"

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
        orig_img1 = image_reader(str(path1))
        orig_img2 = image_reader(str(path2))

        ori_normalizer1 = OrientationNormalizer.create_if_needed(orientation1)
        ori_normalizer2 = OrientationNormalizer.create_if_needed(orientation2)

        orig_img1 = cv2.cvtColor(orig_img1, cv2.COLOR_BGR2RGB)
        orig_img2 = cv2.cvtColor(orig_img2, cv2.COLOR_BGR2RGB)

        if cropper:
            cropper.set_original_image(orig_img1, orig_img2)
            img1, img2 = cropper.crop_ndarray_image(orig_img1, orig_img2)
        else:
            img1 = orig_img1
            img2 = orig_img2

        if ori_normalizer1:
            ori_normalizer1.set_original_image(img1)
            img1 = ori_normalizer1.get_upright_image_ndarray()
        if ori_normalizer2:
            ori_normalizer2.set_original_image(img2)
            img2 = ori_normalizer2.get_upright_image_ndarray()

        assert self.conf.resize
        img1, scale1 = preprocess(img1, resize_max=self.conf.resize.gim_resize)
        img2, scale2 = preprocess(img2, resize_max=self.conf.resize.gim_resize)

        img1 = img1.to(self.device, non_blocking=True)[None]
        img2 = img2.to(self.device, non_blocking=True)[None]

        data = dict(color0=img1, color1=img2, image0=img1, image1=img2)

        with torch.autocast(self.device.type):
            self.model(data)

        mkpts1 = data["mkpts0_f"]
        mkpts2 = data["mkpts1_f"]
        mconf = data["mconf"]

        height1, width1 = img1.shape[-2:]
        height2, width2 = img2.shape[-2:]

        mkpts1 = mkpts1.cpu().numpy()
        mkpts2 = mkpts2.cpu().numpy()
        scores = mconf.cpu().numpy()

        if len(mkpts1) == 0:
            return

        mkpts1[:, 0] *= scale1[0]
        mkpts1[:, 1] *= scale1[1]
        mkpts2[:, 0] *= scale2[0]
        mkpts2[:, 1] *= scale2[1]

        mkpts1, mkpts2, scores = sort_matched_keypoints_by_score(mkpts1, mkpts2, scores)

        if self.conf.confidence_threshold is not None:
            keep = scores >= self.conf.confidence_threshold
            mkpts1 = mkpts1[keep]
            mkpts2 = mkpts2[keep]
            scores = scores[keep]

        if self.conf.topk is not None:
            mkpts1 = mkpts1[: self.conf.topk]
            mkpts2 = mkpts2[: self.conf.topk]
            scores = scores[: self.conf.topk]

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
