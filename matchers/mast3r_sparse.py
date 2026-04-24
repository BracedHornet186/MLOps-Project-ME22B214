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

from scripts.data import FilePath, resolve_model_path
from scripts.extractor import LocalFeatureExtractor
from matchers.base import DetectorFreeMatcher, PointTrackingMatcher
from matchers.config import MASt3RSparseMatcherConfig
from matchers.mast3r import load_images_fixed
from models.mast3r.model import get_mast3r_model
from preprocesses.region import OverlapRegionCropper
from scripts.storage import KeypointStorage, MatchedKeypointStorage, MatchingStorage


def read_image(path: str) -> np.ndarray:
    return cv2.imread(str(path))


class MASt3RSparseMatcher(PointTrackingMatcher):
    def __init__(
        self,
        conf: MASt3RSparseMatcherConfig,
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

        # NOTE: Read images again without caches
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

        keeps = scores >= self.conf.match_threshold
        mkpts1 = mkpts1[keeps]
        mkpts2 = mkpts2[keeps]
        scores = scores[keeps]
        idxs = idxs[keeps]

        order = np.argsort(-scores)
        mkpts1 = mkpts1[order]
        mkpts2 = mkpts2[order]
        scores = scores[order]
        idxs = idxs[order]

        if self.conf.match_topk:
            mkpts1 = mkpts1[: self.conf.match_topk]
            mkpts2 = mkpts2[: self.conf.match_topk]
            scores = scores[: self.conf.match_topk]
            idxs = idxs[: self.conf.match_topk]

        if len(mkpts1) == 0:
            mkpts1 = np.empty((0, 2), dtype=np.float32)
            mkpts2 = np.empty((0, 2), dtype=np.float32)

        if cropper:
            mkpts1, mkpts2 = cropper.convert_cropped_to_original_coordinates(
                mkpts1, mkpts2
            )

        if self.conf.min_matches is None or len(mkpts1) >= self.conf.min_matches:
            matching_storage.add(path1, path2, idxs)
            matched_keypoint_storage.add(path1, path2, mkpts1, mkpts2, scores=scores)


@torch.inference_mode()
def nn_correspondences(
    descs1: torch.Tensor,
    descs2: torch.Tensor,
    scores1: torch.Tensor,  # Shape(H, W)
    scores2: torch.Tensor,  # Shape(H, W)
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    origin_kpts1: np.ndarray,
    origin_kpts2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    H1, W1, _ = descs1.shape
    H2, W2, _ = descs2.shape

    kpts1_tensor = torch.tensor(kpts1, dtype=torch.float32)  # Shape: (N, 2)
    descs1 = descs1.permute(2, 0, 1).unsqueeze(0)
    scores1 = scores1[None, None]
    grid1 = (
        torch.stack(
            [
                2.0 * kpts1_tensor[:, 0] / (W1 - 1) - 1,  # Normalize x
                2.0 * kpts1_tensor[:, 1] / (H1 - 1) - 1,  # Normalize y
            ],
            dim=-1,
        )
        .unsqueeze(0)
        .unsqueeze(2)
        .to(descs1.device)
    )
    descs1 = torch.nn.functional.grid_sample(
        descs1, grid1, align_corners=True, mode="bilinear"
    )  # Shape: (1, C, N, 1)
    scores1 = torch.nn.functional.grid_sample(
        scores1, grid1, align_corners=True, mode="bilinear"
    )  # Shape: (1, 1, N, 1)
    descs1 = descs1.squeeze(0).squeeze(-1).T  # Shape: (N, C)
    scores1 = scores1.squeeze()  # Shape: (N)

    kpts2_tensor = torch.tensor(kpts2, dtype=torch.float32)  # Shape: (N, 2)
    descs2 = descs2.permute(2, 0, 1).unsqueeze(0)
    scores2 = scores2[None, None]
    grid2 = (
        torch.stack(
            [
                2.0 * kpts2_tensor[:, 0] / (W2 - 1) - 1,  # Normalize x
                2.0 * kpts2_tensor[:, 1] / (H2 - 1) - 1,  # Normalize y
            ],
            dim=-1,
        )
        .unsqueeze(0)
        .unsqueeze(2)
        .to(descs2.device)
    )
    descs2 = torch.nn.functional.grid_sample(
        descs2, grid2, align_corners=True, mode="bilinear"
    )  # Shape: (1, C, N, 1)
    scores2 = torch.nn.functional.grid_sample(
        scores2, grid2, align_corners=True, mode="bilinear"
    )  # Shape: (1, 1, N, 1)
    descs2 = descs2.squeeze(0).squeeze(-1).T  # Shape: (N, C)
    scores2 = scores2.squeeze()  # Shape: (N)

    nn1, nn2 = bruteforce_reciprocal_nns(
        descs1,
        descs2,
        device=descs1.device,  # type: ignore
        dist="dot",
        block_size=2**13,
    )
    reciprocal_in_P1 = nn2[nn1] == np.arange(len(nn1))

    scores = scores2[nn1][reciprocal_in_P1].cpu().numpy()
    mkpts1 = origin_kpts1[reciprocal_in_P1]
    mkpts2 = origin_kpts2[nn1][reciprocal_in_P1]

    idx1, *_ = np.where(reciprocal_in_P1)
    idx2 = nn1[reciprocal_in_P1]

    assert len(idx1) == len(idx2)
    idx = np.concatenate([idx1[..., None], idx2[..., None]], axis=1)

    return mkpts1, mkpts2, scores, idx
