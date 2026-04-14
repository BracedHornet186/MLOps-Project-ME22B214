from typing import Callable, Optional

import cv2
import numpy as np
import torch
from dotmap import DotMap

from data import FilePath, resolve_model_path
from matchers.base import DetectorFreeMatcher
from matchers.config import ECOTRConfig
from models.ecotr.config.default import get_cfg_defaults
from models.ecotr.models.ecotr_engines import ECOTR_Engine
from models.ecotr.utils.misc import lower_config
from preprocess import resize_image_opencv
from preprocesses.region import OverlapRegionCropper
from storage import MatchedKeypointStorage
from workspace import log


def read_image(path: str) -> np.ndarray:
    return cv2.imread(str(path))


class ECOTRMatcher(DetectorFreeMatcher):
    def __init__(self, conf: ECOTRConfig, device: Optional[torch.device] = None):
        weight_path = resolve_model_path(conf.weight_path)
        _cfg = get_cfg_defaults()
        _cfg.ECOTR.LOAD_WEIGHTS_PATH = str(weight_path)
        cfg = DotMap(lower_config(_cfg))
        engine = ECOTR_Engine(cfg.ecotr)
        if str(device).startswith("cuda"):
            engine.use_cuda = True  # type: ignore
        engine.load_weight(device)  # type: ignore
        if conf.max_kpts_num:
            engine.MAX_KPTS_NUM = int(conf.max_kpts_num)
        if conf.aspect_ratios:
            engine.ASPECT_RATIOS = list(conf.aspect_ratios)

        self.engine = engine
        self.conf = conf
        self.device = device

        log(f"[ECOTRMatcher] Loaded weights from {weight_path}")
        log(f"[ECOTRMatcher] Use the device ({device})")
        log(f"[ECOTRMatcher] MAX_KPTS_NUM: {engine.MAX_KPTS_NUM}")
        log(f"[ECOTRMatcher] ASPECT_RATIOS: {engine.ASPECT_RATIOS}")

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

        if self.conf.resize:
            img1, scale1, mask1 = resize_image_opencv(
                img1.copy(), conf=self.conf.resize, order3ch="hwc"
            )
            img2, scale2, mask2 = resize_image_opencv(
                img2.copy(), conf=self.conf.resize, order3ch="hwc"
            )
        else:
            scale1, mask1 = np.array([1.0, 1.0]), None
            scale2, mask2 = np.array([1.0, 1.0]), None

        matches = self.engine(img1, img2, cycle=self.conf.cycle, level=self.conf.level)

        # NOTE: Shape(N, 5)
        matches = matches[matches[:, -1] < self.conf.uncertainty_threshold]
        mkpts1 = matches[:, :2].copy()  # xy order
        mkpts2 = matches[:, 2:4].copy()  # xy order
        scores = matches[:, -1].copy()
        if len(mkpts1) == len(mkpts2) == 0:
            mkpts1 = np.empty((0, 2))
            mkpts2 = np.empty((0, 2))
            scores = np.empty((0,))
        else:
            mkpts1[:, 0] *= scale1[0]
            mkpts1[:, 1] *= scale1[1]
            mkpts2[:, 0] *= scale2[0]
            mkpts2[:, 1] *= scale2[1]
        
        if self.conf.min_matches is None or len(mkpts1) >= self.conf.min_matches:
            matched_keypoint_storage.add(path1, path2, mkpts1, mkpts2, scores=scores)
