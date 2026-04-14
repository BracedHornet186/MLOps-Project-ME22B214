from typing import Callable, Optional

import cv2
import numpy as np
import torch

from data import FilePath, resolve_model_path
from matchers.base import DetectorFreeMatcher
from matchers.config import ASpanFormerConfig
from models.aspanformer.aspanformer.aspanformer import ASpanFormer
from models.aspanformer.config.default import get_cfg_defaults
from models.aspanformer.utils.misc import lower_config
from preprocess import resize_image_opencv
from preprocesses.region import OverlapRegionCropper
from storage import MatchedKeypointStorage


def read_image(path: FilePath) -> np.ndarray:
    return cv2.imread(str(path))


class ASpanFormerMatcher(DetectorFreeMatcher):
    def __init__(self, conf: ASpanFormerConfig, device: Optional[torch.device] = None):
        weight_path = resolve_model_path(conf.weight_path)

        cfg = get_cfg_defaults()

        # Copied from https://github.com/apple/ml-aspanformer/blob/main/configs/aspan/outdoor/aspan_test.py
        cfg.ASPAN.COARSE.COARSEST_LEVEL = [36, 36]
        cfg.ASPAN.COARSE.TRAIN_RES = [832, 832]
        cfg.ASPAN.COARSE.TEST_RES = [1152, 1152]
        cfg.ASPAN.MATCH_COARSE.MATCH_TYPE = "dual_softmax"

        cfg.TRAINER.CANONICAL_LR = 8e-3
        cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
        cfg.TRAINER.WARMUP_RATIO = 0.1
        cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]

        # pose estimation
        cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

        cfg.TRAINER.OPTIMIZER = "adamw"
        cfg.TRAINER.ADAMW_DECAY = 0.1
        cfg.ASPAN.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3

        if conf.thr is not None:
            cfg.ASPAN.MATCH_COARSE.THR = conf.thr

        _cfg = lower_config(cfg)
        _loftr_cfg = lower_config(_cfg["aspan"])

        model = ASpanFormer(config=_cfg["aspan"])
        state_dict = torch.load(weight_path, map_location="cpu")["state_dict"]
        msg = model.load_state_dict(state_dict, strict=False)

        print(f"[ASpanFormer] Load {weight_path} as pretrained weights : msg={msg}")
        print(f"[ASpanFormer] Use the device {device}")

        self.model = model.eval().to(device)
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
        img1, img2 = image_reader(str(path1)), image_reader(str(path2))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        assert self.conf.resize
        resized_img1, scale1, mask1 = resize_image_opencv(img1, conf=self.conf.resize)
        resized_img2, scale2, mask2 = resize_image_opencv(img2, conf=self.conf.resize)

        x1 = torch.from_numpy(resized_img1).float()[None][None] / 255.0  # (1, H, W)
        x2 = torch.from_numpy(resized_img2).float()[None][None] / 255.0  # (1, H, W)

        x1 = x1.to(self.device, non_blocking=True)
        x2 = x2.to(self.device, non_blocking=True)

        data = {
            "image0": x1,
            "image1": x2,
            "scale0": torch.tensor(scale1, device=self.device)[None],
            "scale1": torch.tensor(scale2, device=self.device)[None],
        }
        self.model(data)

        mkpts1 = data["mkpts0_f"].cpu().numpy()
        mkpts2 = data["mkpts1_f"].cpu().numpy()
        scores = data["mconf"].cpu().numpy()

        if self.conf.min_matches is None or len(mkpts1) >= self.conf.min_matches:
            matched_keypoint_storage.add(path1, path2, mkpts1, mkpts2, scores=scores)
