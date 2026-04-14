from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.cuda.amp
import tqdm
from dust3r.inference import (
    check_if_same_size,
    collate_with_cat,
    loss_of_one_batch,
    to_cpu,
)
from dust3r.utils.image import (
    ImgNorm,
    _resize_pil_image,
    exif_transpose,
    heif_support_enabled,
    load_images,
)
from mast3r.fast_nn import bruteforce_reciprocal_nns, extract_correspondences_nonsym
from mast3r.model import AsymmetricMASt3R
from PIL import Image

from data import FilePath, resolve_model_path
from extractor import LocalFeatureExtractor
from matchers.base import DetectorFreeMatcher, PointTrackingMatcher
from matchers.config import MASt3RHybridMatcherConfig
from matchers.mast3r import load_images_fixed, postprocess
from matchers.mast3r_sparse import nn_correspondences
from models.mast3r.model import get_mast3r_model
from preprocesses.region import OverlapRegionCropper
from storage import KeypointStorage, MatchedKeypointStorage, MatchingStorage


def read_image(path: str) -> np.ndarray:
    return cv2.imread(str(path))


class MASt3RHybridMatcher(PointTrackingMatcher):
    def __init__(
        self,
        conf: MASt3RHybridMatcherConfig,
        extractors: list[LocalFeatureExtractor],
        device: torch.device | None = None,
    ):
        assert device is not None
        self.conf = conf
        self._extractors = extractors
        self.device = device
        self.size = conf.size

        print(f"MASt3R: size={self.size}")

        self.model = get_mast3r_model(
            resolve_model_path(conf.model.weight_path), self.device
        )

    @property
    def extractors(self) -> list[LocalFeatureExtractor]:
        return self._extractors

    def is_hybrid(self) -> bool:
        return True

    @torch.inference_mode()
    def __call__(
        self,
        path1: str | Path,
        path2: str | Path,
        keypoint_storage: KeypointStorage,
        matching_storage: MatchingStorage,  # For output
        matched_keypoint_storage: MatchedKeypointStorage,  # For output
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

        if cropper:
            orig_img1 = cv2.cvtColor(orig_img1, cv2.COLOR_BGR2RGB)
            orig_img2 = cv2.cvtColor(orig_img2, cv2.COLOR_BGR2RGB)
            cropper.set_original_image(orig_img1, orig_img2)

        paired_images = load_images_fixed(
            [str(path1), str(path2)],
            size=self.size,
            verbose=False,
            image_reader=image_reader,
            cropper=cropper,
        )
        crop_offset1 = paired_images[0].pop("crop_offset")
        crop_offset2 = paired_images[1].pop("crop_offset")
        size_before_crop1 = paired_images[0].pop("size_before_crop")
        size_before_crop2 = paired_images[1].pop("size_before_crop")
        cx1, cy1 = paired_images[0].pop("crop_center")
        cx2, cy2 = paired_images[1].pop("crop_center")
        halfw1, halfh1 = paired_images[0].pop("half_wh")
        halfw2, halfh2 = paired_images[1].pop("half_wh")
        img1 = paired_images[0].pop("img")
        img2 = paired_images[1].pop("img")

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

        # Common block
        with torch.autocast(self.device.type):
            shape1 = torch.tensor(img1.shape[-2:])[None].to(
                self.device, non_blocking=True
            )
            shape2 = torch.tensor(img2.shape[-2:])[None].to(
                self.device, non_blocking=True
            )
            img1 = img1.to(self.device, non_blocking=True)
            img2 = img2.to(self.device, non_blocking=True)

            # compute encoder only once
            feat1, feat2, pos1, pos2 = self.model._encode_image_pairs(
                img1, img2, shape1, shape2
            )

            # decoder 1-2
            dec1, dec2 = self.model._decoder(feat1, pos1, feat2, pos2)
            with torch.autocast(self.device.type, enabled=False):
                pred1 = self.model._downstream_head(
                    1, [tok.float() for tok in dec1], shape1
                )
                pred2 = self.model._downstream_head(
                    2, [tok.float() for tok in dec2], shape2
                )

        desc1, desc2 = (
            pred1["desc"].squeeze(0).detach(),
            pred2["desc"].squeeze(0).detach(),
        )

        conf1, conf2 = (
            pred1["desc_conf"].squeeze(0).detach(),
            pred2["desc_conf"].squeeze(0).detach(),
        )

        # Sparse matching
        # -------------------------
        mkpts1, mkpts2, scores, idxs = nn_correspondences(
            desc1,
            desc2,
            conf1,
            conf2,
            kpts1,
            kpts2,
            origin_kpts1[mask1],
            origin_kpts2[mask2],
        )
        # NOTE
        # `idx` range is corresponding to keypoints after masking.
        # Thus, remap it to original keypoint range
        used_origin_idx1, *_ = np.where(mask1)
        used_origin_idx2, *_ = np.where(mask2)
        idxs[:, 0] = used_origin_idx1[idxs[:, 0]]
        idxs[:, 1] = used_origin_idx2[idxs[:, 1]]

        keeps = scores >= self.conf.sparse_match_threshold
        mkpts1 = mkpts1[keeps]
        mkpts2 = mkpts2[keeps]
        scores = scores[keeps]
        idxs = idxs[keeps]

        order = np.argsort(-scores)
        mkpts1 = mkpts1[order]
        mkpts2 = mkpts2[order]
        scores = scores[order]
        idxs = idxs[order]

        if self.conf.sparse_match_topk:
            mkpts1 = mkpts1[: self.conf.sparse_match_topk]
            mkpts2 = mkpts2[: self.conf.sparse_match_topk]
            scores = scores[: self.conf.sparse_match_topk]
            idxs = idxs[: self.conf.sparse_match_topk]

        if len(mkpts1) == 0:
            mkpts1 = np.empty((0, 2), dtype=np.float32)
            mkpts2 = np.empty((0, 2), dtype=np.float32)

        if cropper:
            mkpts1, mkpts2 = cropper.convert_cropped_to_original_coordinates(
                mkpts1, mkpts2
            )

        num_sparse_matches = len(mkpts1)
        if (
            self.conf.sparse_min_matches is None
            or num_sparse_matches >= self.conf.sparse_min_matches
        ):
            matching_storage.add(path1, path2, idxs)
        else:
            num_sparse_matches = 0

        prefer_mask1 = None
        prefer_mask2 = None
        if self.conf.prefer_sparse_matches and num_sparse_matches > 0:
            prefer_mask1 = np.zeros((orig_H1, orig_W1), dtype=np.uint8)
            prefer_mask2 = np.zeros((orig_H2, orig_W2), dtype=np.uint8)
            x1 = mkpts1[:, 0].copy()
            y1 = mkpts1[:, 1].copy()
            x2 = mkpts2[:, 0].copy()
            y2 = mkpts2[:, 1].copy()
            x1 = np.clip(x1, 0, orig_W1 - 1).astype(np.int64)
            y1 = np.clip(y1, 0, orig_H1 - 1).astype(np.int64)
            x2 = np.clip(x2, 0, orig_W2 - 1).astype(np.int64)
            y2 = np.clip(y2, 0, orig_H2 - 1).astype(np.int64)
            prefer_mask1[y1, x1] = 1
            prefer_mask2[y2, x2] = 1
            kernel = np.ones(
                (
                    self.conf.prefer_sparse_kernel_size,
                    self.conf.prefer_sparse_kernel_size,
                ),
                dtype=np.uint8,
            )
            prefer_mask1 = cv2.dilate(prefer_mask1, kernel)
            prefer_mask2 = cv2.dilate(prefer_mask2, kernel)

        # Semi-dense matching
        # -------------------------
        corres: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (
            extract_correspondences_nonsym(
                desc1,
                desc2,
                conf1.cpu(),  # NOTE: conf on device raises an error
                conf2.cpu(),
                device=self.device,
                subsample=self.conf.dense_subsample,
                pixel_tol=self.conf.dense_pixel_tol,
            )
        )  # type: ignore
        score = corres[2]
        mask = score >= self.conf.dense_match_threshold
        mkpts1 = corres[0][mask].cpu().numpy()
        mkpts2 = corres[1][mask].cpu().numpy()
        scores = score[mask].cpu().numpy()

        order = np.argsort(-scores)
        mkpts1 = mkpts1[order]
        mkpts2 = mkpts2[order]
        scores = scores[order]

        if self.conf.dense_match_topk:
            mkpts1 = mkpts1[: self.conf.dense_match_topk]
            mkpts2 = mkpts2[: self.conf.dense_match_topk]
            scores = scores[: self.conf.dense_match_topk]

        if len(mkpts1) == 0:
            mkpts1 = np.empty((0, 2), dtype=np.float32)
            mkpts2 = np.empty((0, 2), dtype=np.float32)
        else:
            mkpts1 = postprocess(
                mkpts1,
                orig_img1,
                paired_images[0]["true_shape"],
                crop_offset1,
                size_before_crop1,
            )
            mkpts2 = postprocess(
                mkpts2,
                orig_img2,
                paired_images[1]["true_shape"],
                crop_offset2,
                size_before_crop2,
            )

        if cropper:
            mkpts1, mkpts2 = cropper.convert_cropped_to_original_coordinates(
                mkpts1, mkpts2
            )

        if self.conf.prefer_sparse_matches and num_sparse_matches > 0:
            assert isinstance(prefer_mask1, np.ndarray)
            assert isinstance(prefer_mask2, np.ndarray)
            _mkpts1 = mkpts1.copy()
            _mkpts2 = mkpts2.copy()
            _mkpts1 = _mkpts1.astype(np.int64)
            _mkpts2 = _mkpts2.astype(np.int64)
            _mkpts1[:, 0] = np.clip(_mkpts1[:, 0], 0, orig_W1 - 1)
            _mkpts1[:, 1] = np.clip(_mkpts1[:, 1], 0, orig_H1 - 1)
            _mkpts2[:, 0] = np.clip(_mkpts2[:, 0], 0, orig_W2 - 1)
            _mkpts2[:, 1] = np.clip(_mkpts2[:, 1], 0, orig_H2 - 1)
            keeps1 = prefer_mask1[_mkpts1[:, 1], _mkpts1[:, 0]] == 0
            keeps2 = prefer_mask2[_mkpts2[:, 1], _mkpts2[:, 0]] == 0
            keeps = keeps1 & keeps2
            mkpts1 = mkpts1[keeps]
            mkpts2 = mkpts2[keeps]

        num_dense_matches = len(mkpts1)
        if (
            self.conf.dense_min_matches is None
            or num_dense_matches >= self.conf.dense_min_matches
        ):
            matched_keypoint_storage.add(path1, path2, mkpts1, mkpts2, scores=scores)
        else:
            num_dense_matches = 0

        if False:
            print(
                f"MASt3RHybridMatcher | ({path1}, {path2}): "
                f"{num_dense_matches=}, {num_sparse_matches=}"
            )
