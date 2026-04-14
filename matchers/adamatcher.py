from collections import defaultdict
from typing import Callable, Optional, Tuple

import cv2
import kornia
import numpy as np
import torch
from dotmap import DotMap
from yacs.config import CfgNode as CN

from data import FilePath, resolve_model_path
from matchers.base import DetectorFreeMatcher
from matchers.config import AdaMatcherConfig
from models.adamatcher.adamatcher.adamatcher import AdaMatcher
from models.adamatcher.config.default import get_cfg_defaults
from models.adamatcher.utils.misc import lower_config
from preprocess import resize_image_opencv
from preprocesses.config import ResizeConfig
from preprocesses.orientation import OrientationNormalizer
from preprocesses.region import OverlapRegionCropper
from postprocesses.nms import sort_matched_keypoints_by_score
from storage import MatchedKeypointStorage, MatchingStorage
from workspace import log


def read_image(path: FilePath) -> np.ndarray:
    return cv2.imread(str(path))


class AdaMatcherMatcher(DetectorFreeMatcher):
    def __init__(self, conf: AdaMatcherConfig, device: Optional[torch.device] = None):
        weight_path = resolve_model_path(conf.weight_path)

        cfg = get_cfg_defaults()
        if conf.weight_path == "ADAMATCHER":
            # loftr_ds_dense.py
            cfg.ADAMATCHER.MATCH_COARSE.MATCH_TYPE = "dual_softmax"
            cfg.ADAMATCHER.MATCH_COARSE.SPARSE_SPVS = False

            cfg.TRAINER.WARMUP_TYPE = "linear"  # [linear, constant]
            cfg.TRAINER.CANONICAL_LR = 8e-3
            cfg.TRAINER.WARMUP_STEP = 1875
            cfg.TRAINER.WARMUP_RATIO = 0.1
            cfg.TRAINER.MSLR_GAMMA = 0.5
            cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]

            # pose estimation
            cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

            cfg.ADAMATCHER.RESOLUTION = (32, 8, 2)  # (16,8,2)
            cfg.DATASET.MGDPT_DF = cfg.ADAMATCHER.RESOLUTION[0]
            cfg.ADAMATCHER.MATCH_COARSE.T_K = -1

            cfg.TRAINER.OPTIMIZER = "adamw"
            cfg.TRAINER.ADAMW_DECAY = 0.1
            cfg.ADAMATCHER.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
        else:
            raise ValueError(conf.weight_path)

        if conf.thr is not None:
            cfg.ADAMATCHER.MATCH_COARSE.THR = conf.thr

        _cfg = lower_config(cfg)
        self.ada_cfg = lower_config(_cfg["adamatcher"])

        model = AdaMatcher(config=_cfg["adamatcher"])
        weights = torch.load(weight_path, map_location="cpu")["state_dict"]
        weights = {
            k[8:]: v for k, v in weights.items()
        }  # Remove prefix 'matcher.' of keys
        model.load_state_dict(weights)

        log(f"[AdaMatcherMatcher] Load {weight_path} as pretrained checkpoint")
        log(f"[AdaMatcherMatcher] Use the device ({device})")

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
        # NOTE
        # preprocess code: https://github.com/TencentYoutuResearch/AdaMatcher/blob/main/src/utils/dataset.py#L156
        img1, img2 = image_reader(str(path1)), image_reader(str(path2))

        ori_normalizer1 = OrientationNormalizer.create_if_needed(orientation1)
        ori_normalizer2 = OrientationNormalizer.create_if_needed(orientation2)

        if cropper:
            cropper.set_original_image(img1, img2)
            img1, img2 = cropper.crop_ndarray_image(img1, img2)

        if ori_normalizer1:
            ori_normalizer1.set_original_image(img1)
            img1 = ori_normalizer1.get_upright_image_ndarray()
        if ori_normalizer2:
            ori_normalizer2.set_original_image(img2)
            img2 = ori_normalizer2.get_upright_image_ndarray()

        x1, scale1, mask1 = preprocess(
            img1, resize=self.conf.resize, device=self.device
        )
        x2, scale2, mask2 = preprocess(
            img2, resize=self.conf.resize, device=self.device
        )
        h1, w1 = x1.shape[2:]
        h2, w2 = x1.shape[2:]

        data = {
            "image0": x1,  # Shape(1, C, H, W)
            "image1": x2,  # Shape(1, C, H, W)
            "scale0": scale1,
            "scale1": scale2,
            "scale_wh0": torch.tensor([w1, h1])[None],
            "scale_wh1": torch.tensor([w2, h2])[None],
        }
        self.model(data)

        mkpts1 = data["mkpts0_f"].cpu().numpy()
        mkpts2 = data["mkpts1_f"].cpu().numpy()
        scores = data["scores"].cpu().numpy()
        # mkpts1 = postprocess(data['mkpts0_f'], img1, scale1, mask1)
        # mkpts2 = postprocess(data['mkpts1_f'], img2, scale2, mask2)

        # if self.conf.confidence_threshold is not None:
        #    scores = data['mconf'].cpu().numpy()
        #    keeps = scores > self.conf.confidence_threshold
        #    mkpts1 = mkpts1[keeps, :]
        #    mkpts2 = mkpts2[keeps, :]

        mkpts1, mkpts2, scores = sort_matched_keypoints_by_score(mkpts1, mkpts2, scores)

        if self.conf.confidence_threshold is not None:
            keeps = scores > self.conf.confidence_threshold
            mkpts1 = mkpts1[keeps, :]
            mkpts2 = mkpts2[keeps, :]
            scores = scores[keeps]

        if self.conf.topk is not None:
            k = self.conf.topk
            mkpts1 = mkpts1[:k]
            mkpts2 = mkpts2[:k]
            scores = scores[:k]

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
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize:
        img, scale, mask = resize_image_opencv(img, conf=resize, order3ch="hwc")

        # ToTensor and Normalize
        x = torch.from_numpy(img).float() / 255.0
        x = x.permute(2, 0, 1)  # Shape(C, H, W)
        x = x.unsqueeze(0)  # Shape(1, C, H, W)
        scale = torch.from_numpy(scale)[None]
        if mask is not None:
            mask = torch.from_numpy(mask).bool()[None]  # (h, w) -> (1, h, w)
    else:
        raise NotImplementedError

    if device:
        x = x.to(device, non_blocking=True)
        scale = scale.to(device, non_blocking=True)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
    return x, scale, mask


def postprocess(
    kpts: torch.Tensor,
    img: np.ndarray,
    scale: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    _kpts = kpts.cpu().numpy().astype(np.float32)
    _kpts[:, 0] *= scale[0, 0].float().item()
    _kpts[:, 1] *= scale[0, 1].float().item()
    return _kpts
