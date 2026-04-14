import copy
from typing import Callable, Optional, Tuple

import cv2
import kornia
import numpy as np
import torch
from yacs.config import CfgNode as CN

from data import FilePath, resolve_model_path
from matchers.base import DetectorFreeMatcher
from matchers.config import EfficientLoFTRConfig
from models.efficientloftr.config.default import get_cfg_defaults
from models.efficientloftr.loftr import full_default_cfg, reparameter
from models.efficientloftr.loftr.loftr import LoFTR
from preprocess import paired_pre_resize, resize_image_opencv
from preprocesses.config import ResizeConfig
from preprocesses.orientation import OrientationNormalizer
from preprocesses.region import OverlapRegionCropper
from storage import MatchedKeypointStorage
from workspace import log


def read_image(path: FilePath) -> np.ndarray:
    return cv2.imread(str(path))


class EfficientLoFTRMatcher(DetectorFreeMatcher):
    def __init__(self, conf: EfficientLoFTRConfig, device: Optional[torch.device] = None):
        model, cfg = load_model(conf)
        model = model.eval().to(device)
        log(f"[EfficientLoFTR] model loaded weights from {conf.weight_path}")
        log(f"[EfficientLoFTR] Use the device ({device})")
        self.model = model
        self.cfg = cfg
        self.conf = conf
        self.device = device
        assert conf.resize
        assert conf.resize.pad_bottom_right
        assert conf.resize.divisible_factor == 8

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

        if self.conf.paired_pre_resize:
            pre_resized_img1, pre_resized_img2 = paired_pre_resize(
                img1, img2, conf=self.conf.paired_pre_resize
            )
        else:
            pre_resized_img1, pre_resized_img2 = img1, img2

        x1, scale1, mask1 = preprocess(
            pre_resized_img1, resize=self.conf.resize, device=self.device
        )
        x2, scale2, mask2 = preprocess(
            pre_resized_img2, resize=self.conf.resize, device=self.device
        )

        data = {
            "image0": x1,
            "image1": x2,
            #'depth0': torch.tensor([]).to(x1.device, non_blocking=True),
            #'depth1': torch.tensor([]).to(x2.device, non_blocking=True),
            #'scale0': scale1,
            #'scale1': scale2,
        }

        self.model(data)

        mkpts1: np.ndarray = data["mkpts0_f"].cpu().numpy()
        mkpts2: np.ndarray = data["mkpts1_f"].cpu().numpy()
        scores: np.ndarray = data["mconf"].cpu().numpy()

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

        if len(mkpts1) == 0:
            return

        mkpts1 = postprocess(mkpts1, img1, pre_resized_img1, scale1, mask1)
        mkpts2 = postprocess(mkpts2, img2, pre_resized_img2, scale2, mask2)

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


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def load_model(conf: EfficientLoFTRConfig) -> Tuple[LoFTR, CN]:
    #cfg = get_cfg_defaults()
    cfg = copy.deepcopy(full_default_cfg)
    if conf.config_file_path:
        config_file_path = resolve_model_path(conf.config_file_path)
        print(f"Merging {config_file_path}")
        cfg.merge_from_file(config_file_path)

    if conf.match_coarse_thr is not None:
        cfg.LOFTR.MATCH_COARSE.THR = conf.match_coarse_thr

    #_cfg = lower_config(cfg)

    matcher = LoFTR(cfg)

    if conf.weight_path:
        weight_path = resolve_model_path(conf.weight_path)
        state_dict = torch.load(weight_path, map_location="cpu")["state_dict"]
        print(f"Load '{weight_path}' as pretrained checkpoint")
        matcher.load_state_dict(state_dict)

    matcher = reparameter(matcher)
    matcher = matcher.cuda().eval()
    return matcher, cfg


def preprocess(
    img: np.ndarray,
    resize: Optional[ResizeConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if resize:
        img, scale, mask = resize_image_opencv(img, conf=resize)

        # ToTensor and Normalize
        x = (
            torch.from_numpy(img).float()[None][None] / 255
        )  # (h, w) -> (1, 1, h, w) normalized
        scale = torch.from_numpy(scale)[None]
        if mask is not None:
            mask = torch.from_numpy(mask).bool()[None]  # (h, w) -> (1, h, w)
    else:
        x = kornia.utils.image_to_tensor(img, keepdim=False).float() / 255.0
        scale = torch.tensor([1.0, 1.0], dtype=torch.float)[None]
        mask = None

    if device:
        x = x.to(device, non_blocking=True)
        scale = scale.to(device, non_blocking=True)
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
    return x, scale, mask


def postprocess(
    kpts: np.ndarray,
    img: np.ndarray,
    pre_resized_img: np.ndarray,
    scale: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    # Rescale: network input -> pre resized image
    kpts[:, 0] *= scale[0, 0].item()
    kpts[:, 1] *= scale[0, 1].item()

    # Rescale: pre resized image -> source image
    orig_H, orig_W = img.shape[:2]
    pre_resized_H, pre_resized_W = pre_resized_img.shape[:2]
    scale_H = orig_H / pre_resized_H
    scale_W = orig_W / pre_resized_W
    kpts[:, 0] *= scale_W
    kpts[:, 1] *= scale_H

    return kpts
