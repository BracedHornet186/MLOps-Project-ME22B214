from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import kornia
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
import torch
from kornia_moons.feature import draw_LAF_matches
from PIL import Image

from scripts.data import FilePath
from data_schema import DataSchema
from scripts.extractor import LocalFeatureExtractor
from matchers.base import DetectorFreeMatcher, LocalFeatureMatcher, PointTrackingMatcher
from pipelines.config import RANSACConfig
from pipelines.scene import Scene
from pipelines.verification import run_ransac
from scripts.storage import (
    InMemoryKeypointStorage,
    InMemoryLocalFeatureStorage,
    InMemoryMatchedKeypointStorage,
    InMemoryMatchingStorage,
    fuse_matching_sets_late,
)

plt.rcParams["figure.dpi"] = 200


class LocalFeatureMatchingVisualizer:
    def __init__(self, extractor: LocalFeatureExtractor, matcher: LocalFeatureMatcher):
        self.extractor = extractor
        self.matcher = matcher

    def __call__(
        self,
        path1: FilePath,
        path2: FilePath,
        ransac_conf: RANSACConfig | None = None,
        print_keypoints: bool = False,
        max_matches: int | None = None,
    ):
        m_storage = InMemoryMatchingStorage()
        f_storage = InMemoryLocalFeatureStorage()

        img1 = cv2.imread(str(path1))
        img2 = cv2.imread(str(path2))
        shape1 = img1.shape[:2]
        shape2 = img2.shape[:2]

        self.extractor(path1, storage=f_storage)
        self.extractor(path2, storage=f_storage)

        self.matcher(path1, path2, shape1, shape2, f_storage, m_storage)
        idxs = m_storage.get(path1, path2)

        _, kpts1, _, _ = f_storage.get(path1)
        _, kpts2, _, _ = f_storage.get(path2)
        mkpts1 = kpts1[idxs[:, 0]]
        mkpts2 = kpts2[idxs[:, 1]]

        inliers = None
        if ransac_conf:
            F, inliers = run_ransac(
                mkpts1, mkpts2, ransac_conf, min_matches_required=16
            )

        if print_keypoints:
            print(mkpts1)
            print(mkpts2)

        print(f"(path1, path2) = ({path1}, {path2})")
        print(f"(#kpts1, #kpts2) = ({len(kpts1)}, {len(kpts2)})")
        print(f"#matches = {len(mkpts1)}")
        if inliers is not None:
            print(f"#inliers = {sum(inliers)}")

        if max_matches:
            print(f"Visualize matches up to {max_matches}")
            mkpts1 = mkpts1[:max_matches]
            mkpts2 = mkpts2[:max_matches]

        draw(path1, path2, mkpts1, mkpts2, inliers=inliers)


class DetectorFreeMatchingVisualizer:
    def __init__(self, matcher: DetectorFreeMatcher):
        self.matcher = matcher

    def __call__(
        self,
        path1: FilePath,
        path2: FilePath,
        ransac_conf: Optional[RANSACConfig] = None,
        max_matches: int | None = None,
    ):
        mkpt_storage = InMemoryMatchedKeypointStorage()

        self.matcher(path1, path2, mkpt_storage)

        # TODO
        mkpts1, mkpts2 = mkpt_storage.get(path1, path2)
        inliers = None
        if ransac_conf:
            F, inliers = run_ransac(
                mkpts1, mkpts2, ransac_conf, min_matches_required=16
            )

        print(f"(path1, path2) = ({path1}, {path2})")
        print(f"#matches = {len(mkpts1)}")
        if inliers is not None:
            print(f"#inliers = {sum(inliers)}")

        if max_matches:
            print(f"Visualize matches up to {max_matches}")
            mkpts1 = mkpts1[:max_matches]
            mkpts2 = mkpts2[:max_matches]

        draw(path1, path2, mkpts1, mkpts2, inliers=inliers)


class PointTrackingMatchingVisualizer:
    def __init__(self, matcher: PointTrackingMatcher):
        self.matcher = matcher

    def __call__(
        self,
        path1: FilePath,
        path2: FilePath,
        data_schema: DataSchema,
        ransac_conf: RANSACConfig | None = None,
        max_matches: int | None = None,
        random_sampling: bool = False,
    ):
        scene = Scene(
            "test_dataset",
            "test_scene",
            [path1, path2],
            Path(path1).parent,
            data_schema,
        )
        k_storage = self.matcher.extract_keypoints(scene)
        m_storage = InMemoryMatchingStorage()
        mk_storage = InMemoryMatchedKeypointStorage()
        self.matcher(path1, path2, k_storage, m_storage, mk_storage)

        if True:
            if self.matcher.is_hybrid():
                _k_storage, _m_storage = mk_storage.to_keypoints_and_matches()
                concat_k_storage, concat_m_storage = fuse_matching_sets_late(
                    [(k_storage, m_storage), (_k_storage, _m_storage)],
                    scene,
                )
                kpts1 = concat_k_storage.get(path1)
                kpts2 = concat_k_storage.get(path2)
                idxs = concat_m_storage.get(path1, path2)
                mkpts1 = kpts1[idxs[:, 0]]
                mkpts2 = kpts2[idxs[:, 1]]
            else:
                # Use keypoints and matchings
                kpts1 = k_storage.get(path1)
                kpts2 = k_storage.get(path2)
                idxs = m_storage.get(path1, path2)
                mkpts1 = kpts1[idxs[:, 0]]
                mkpts2 = kpts2[idxs[:, 1]]
        else:
            # Use matched keypoints
            mkpts1, mkpts2 = mk_storage.get(path1, path2)

        inliers = None
        if ransac_conf:
            F, inliers = run_ransac(
                mkpts1, mkpts2, ransac_conf, min_matches_required=16
            )

        print(f"(path1, path2) = ({path1}, {path2})")
        print(f"#matches = {len(mkpts1)}")
        if inliers is not None:
            print(f"#inliers = {sum(inliers)}")

        if max_matches:
            print(f"Visualize matches up to {max_matches}")
            if random_sampling:
                random_idx = np.random.permutation(len(mkpts1))[:max_matches]
                mkpts1 = mkpts1[random_idx]
                mkpts2 = mkpts2[random_idx]
            else:
                mkpts1 = mkpts1[:max_matches]
                mkpts2 = mkpts2[:max_matches]

        draw(path1, path2, mkpts1, mkpts2, inliers=inliers)


def draw(
    path1: FilePath,
    path2: FilePath,
    mkpts1: np.ndarray,
    mkpts2: np.ndarray,
    inliers: Optional[np.ndarray] = None,
    bbox1: Optional[np.ndarray] = None,
    bbox2: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
):
    if ax is None:
        fig, ax = plt.subplots()

    draw_dict = {
        "inlier_color": None,
        "tentative_color": None,
        "feature_color": None,
        "vertical": False,
    }
    draw_dict["inlier_color"] = (0.2, 1, 0.2)
    draw_dict["tentative_color"] = (1, 0.2, 0.2)
    draw_dict["feature_color"] = (0.2, 0.5, 1)

    pil_image1 = Image.open(path1).convert("RGB")
    pil_image2 = Image.open(path2).convert("RGB")
    image1 = np.array(pil_image1)
    image2 = np.array(pil_image2)
    if inliers is None:
        inliers = np.ones((len(mkpts1),)).astype(bool)

    if bbox1 is not None:
        image1 = cv2.rectangle(
            image1,
            pt1=(int(bbox1[0]), int(bbox1[1])),
            pt2=(int(bbox1[2]), int(bbox1[3])),
            color=(0, 255, 0),
            thickness=3,
        )
    if bbox2 is not None:
        image2 = cv2.rectangle(
            image2,
            pt1=(int(bbox2[0]), int(bbox2[1])),
            pt2=(int(bbox2[2]), int(bbox2[3])),
            color=(0, 255, 0),
            thickness=3,
        )

    draw_LAF_matches(
        kornia.feature.laf_from_center_scale_ori(
            torch.from_numpy(mkpts1).view(1, -1, 2),
            torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
            torch.ones(mkpts1.shape[0]).view(1, -1, 1),
        ),
        kornia.feature.laf_from_center_scale_ori(
            torch.from_numpy(mkpts2).view(1, -1, 2),
            torch.ones(mkpts2.shape[0]).view(1, -1, 1, 1),
            torch.ones(mkpts2.shape[0]).view(1, -1, 1),
        ),
        torch.arange(mkpts1.shape[0]).view(-1, 1).repeat(1, 2),
        image1,
        image2,
        inliers,
        draw_dict=draw_dict,
        ax=ax,
    )


def draw_img(
    img1: Image.Image,
    img2: Image.Image,
    mkpts1: np.ndarray,
    mkpts2: np.ndarray,
    inliers: Optional[np.ndarray] = None,
    bbox1: Optional[np.ndarray] = None,
    bbox2: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
):
    if ax is None:
        fig, ax = plt.subplots()

    draw_dict = {
        "inlier_color": None,
        "tentative_color": None,
        "feature_color": None,
        "vertical": False,
    }
    draw_dict["inlier_color"] = (0.2, 1, 0.2)
    draw_dict["tentative_color"] = (1, 0.2, 0.2)
    draw_dict["feature_color"] = (0.2, 0.5, 1)

    pil_image1 = img1
    pil_image2 = img2
    image1 = np.array(pil_image1)
    image2 = np.array(pil_image2)
    if inliers is None:
        inliers = np.ones((len(mkpts1),)).astype(bool)

    if bbox1 is not None:
        image1 = cv2.rectangle(
            image1,
            pt1=(int(bbox1[0]), int(bbox1[1])),
            pt2=(int(bbox1[2]), int(bbox1[3])),
            color=(0, 255, 0),
            thickness=3,
        )
    if bbox2 is not None:
        image2 = cv2.rectangle(
            image2,
            pt1=(int(bbox2[0]), int(bbox2[1])),
            pt2=(int(bbox2[2]), int(bbox2[3])),
            color=(0, 255, 0),
            thickness=3,
        )

    draw_LAF_matches(
        kornia.feature.laf_from_center_scale_ori(
            torch.from_numpy(mkpts1).view(1, -1, 2),
            torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
            torch.ones(mkpts1.shape[0]).view(1, -1, 1),
        ),
        kornia.feature.laf_from_center_scale_ori(
            torch.from_numpy(mkpts2).view(1, -1, 2),
            torch.ones(mkpts2.shape[0]).view(1, -1, 1, 1),
            torch.ones(mkpts2.shape[0]).view(1, -1, 1),
        ),
        torch.arange(mkpts1.shape[0]).view(-1, 1).repeat(1, 2),
        image1,
        image2,
        inliers,
        draw_dict=draw_dict,
        ax=ax,
    )
