from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from mast3r.model import AsymmetricMASt3R
from mpsfm.extraction.pairwise.models.utils.featuremap import NNs_sparse

from scripts.data import resolve_model_path
from extractor import LocalFeatureExtractor, extract_all
from matchers.base import PointTrackingMatcher
from matchers.config import MASt3RMPSFMSparseMatcherConfig
from matchers.mast3r import load_images_fixed
from models.mast3r.model import get_mast3r_model
from preprocesses.region import OverlapRegionCropper
from storage import KeypointStorage, MatchedKeypointStorage, MatchingStorage


class MASt3RMPSFMSparseMatcher(PointTrackingMatcher):
    def __init__(
        self,
        conf: MASt3RMPSFMSparseMatcherConfig,
        extractors: list[LocalFeatureExtractor],
        device: torch.device,
    ):
        self.conf = conf
        self._extractors = extractors
        self.model = get_mast3r_model(
            str(resolve_model_path(conf.model.weight_path)), device
        )
        self.device = device

    @property
    def extractors(self) -> list[LocalFeatureExtractor]:
        return self._extractors

    @torch.inference_mode()
    def __call__(
        self,
        path1: str | Path,
        path2: str | Path,
        keypoint_storage: KeypointStorage,
        matching_storage: MatchingStorage,
        matched_keypoint_storage: MatchedKeypointStorage,
        cropper: OverlapRegionCropper | None = None,
        orientation1: int | None = None,
        orientation2: int | None = None,
        image_reader: Callable[..., Any] | None = None,
    ):
        image_reader = image_reader or read_image

        kpts1 = keypoint_storage.get(path1)
        kpts2 = keypoint_storage.get(path2)

        origin_kpts1 = kpts1.copy()
        origin_kpts2 = kpts2.copy()

        orig_img1 = image_reader(path1)
        orig_img2 = image_reader(path2)

        orig_H1, orig_W1 = orig_img1.shape[:2]
        orig_H2, orig_W2 = orig_img2.shape[:2]

        paired_images = load_images_fixed(
            [str(path1), str(path2)], size=self.conf.size, verbose=False
        )
        crop_offset1 = paired_images[0].pop("crop_offset")
        crop_offset2 = paired_images[1].pop("crop_offset")
        size_before_crop1 = paired_images[0].pop("size_before_crop")
        size_before_crop2 = paired_images[1].pop("size_before_crop")
        cx1, cy1 = paired_images[0].pop("crop_center")
        cx2, cy2 = paired_images[1].pop("crop_center")
        halfw1, halfh1 = paired_images[0].pop("half_wh")
        halfw2, halfh2 = paired_images[1].pop("half_wh")
        im1 = paired_images[0].pop("img")
        im2 = paired_images[1].pop("img")

        # Transform keypoints to resized image coordinates
        kpts1[:, 0] = (kpts1[:, 0] / orig_W1) * size_before_crop1[0]
        kpts1[:, 1] = (kpts1[:, 1] / orig_H1) * size_before_crop1[1]
        kpts2[:, 0] = (kpts2[:, 0] / orig_W2) * size_before_crop2[0]
        kpts2[:, 1] = (kpts2[:, 1] / orig_H2) * size_before_crop2[1]

        mask1 = (
            (crop_offset1[0] <= kpts1[:, 0])
            & (kpts1[:, 0] < (cx1 + halfw1))
            & (crop_offset1[1] <= kpts1[:, 1])
            & (kpts1[:, 1] < (cy1 + halfh1))
        )
        mask2 = (
            (crop_offset2[0] <= kpts2[:, 0])
            & (kpts2[:, 0] < (cx2 + halfw2))
            & (crop_offset2[1] <= kpts2[:, 1])
            & (kpts2[:, 1] < (cy2 + halfh2))
        )

        kpts1 = kpts1[mask1] - np.array([crop_offset1[0], crop_offset1[1]])
        kpts2 = kpts2[mask2] - np.array([crop_offset2[0], crop_offset2[1]])

        with torch.autocast(self.device.type):
            res = symmetric_inference(self.model, im1, im2, self.device)
        X11, _, X22, _ = [r["pts3d"][0].cpu().numpy() for r in res]
        C11, _, C22, _ = [r["conf"][0].cpu().numpy() for r in res]
        descs = [r["desc"][0] for r in res]
        qonfs = [r["desc_conf"][0] for r in res]
        pred = {}

        matches, scores = [
            el[0]
            for el in extract_correspondences_sparse(
                descs,
                qonfs,
                kpts1,
                kpts2,
                subsample=self.conf.subsample,
                scores_thresh=self.conf.nn_score_threshold,
            )
        ]

        # NOTE
        # matches: np.ndarray, Shape(#kpts1,)
        # scores: np.ndarray, Shape(#kpts1,)

        keeps = matches >= 0
        mkpts1 = origin_kpts1[mask1][keeps]
        matches = matches[keeps]
        scores = scores[keeps]

        keeps = scores > self.conf.score_threshold
        mkpts1 = mkpts1[keeps]
        matches = matches[keeps]
        scores = scores[keeps]

        if len(matches) == 0:
            mkpts1 = np.empty((0, 2), dtype=np.float32)
            mkpts2 = np.empty((0, 2), dtype=np.float32)
        else:
            orders = np.argsort(-scores)
            matches = matches[orders]
            scores = scores[orders]
            mkpts1 = mkpts1[orders]
            mkpts2 = origin_kpts2[mask2][matches]
            assert len(mkpts1) == len(mkpts2)

        if self.conf.min_matches is None or len(mkpts1) >= self.conf.min_matches:
            matched_keypoint_storage.add(path1, path2, mkpts1, mkpts2, scores=scores)


def read_image(path: str) -> np.ndarray:
    return cv2.imread(str(path))


def symmetric_inference(
    model: AsymmetricMASt3R,
    img1: torch.Tensor,
    img2: torch.Tensor,
    device: torch.device,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    shape1 = torch.tensor(img1.shape[-2:])[None].to(device, non_blocking=True)
    shape2 = torch.tensor(img2.shape[-2:])[None].to(device, non_blocking=True)
    img1 = img1.to(device, non_blocking=True)
    img2 = img2.to(device, non_blocking=True)

    # compute encoder only once
    feat1, feat2, pos1, pos2 = model._encode_image_pairs(img1, img2, shape1, shape2)

    # decoder 1-2
    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    # decoder 2-1
    res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape2, shape1)

    return (res11, res21, res22, res12)


def decoder(
    model: AsymmetricMASt3R,
    feat1: torch.Tensor,
    feat2: torch.Tensor,
    pos1: torch.Tensor,
    pos2: torch.Tensor,
    shape1: torch.Tensor,
    shape2: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


def extract_correspondences_sparse(
    feats: list[torch.Tensor],
    qonfs: list[torch.Tensor],
    kps0: np.ndarray,
    kps1: np.ndarray,
    subsample: int = 8,
    scores_thresh: float = 0.85,
    ptmap_key: str = "pred_desc",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    feat11, feat21, feat22, feat12 = feats
    qonf11, qonf21, qonf22, qonf12 = qonfs
    assert feat11.shape[:2] == feat12.shape[:2] == qonf11.shape == qonf12.shape
    assert feat21.shape[:2] == feat22.shape[:2] == qonf21.shape == qonf22.shape
    opt = dict(workers=32) if "3d" in ptmap_key else dict(dist="dot", block_size=2**13)
    matches = []
    scores = []

    for A, B, QA, QB in [
        (feat11, feat21, qonf11, qonf21),
        (feat12, feat22, qonf12, qonf22),
    ]:
        matches12, scores12 = NNs_sparse(
            A,
            B,
            QA,
            QB,
            kps0,
            kps1,
            subsample_or_initxy1=subsample,
            ret_xy=False,
            scores_thresh=scores_thresh,
            **opt,
        )
        matches.append(matches12)
        scores.append(scores12)
        break

    return matches, scores
