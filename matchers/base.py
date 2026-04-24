from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import tqdm

from scripts.data import FilePath
from scripts.extractor import LocalFeatureExtractor, extract_all
from pipelines.common import Scene
from postprocesses.panet import PANetRefiner
from preprocesses.region import Cropper, OverlapRegionCropper, OverlapRegionEstimator
from scripts.storage import (
    InMemoryLocalFeatureStorage,
    KeypointStorage,
    Line2DFeatureStorage,
    LocalFeatureStorage,
    MatchedKeypointStorage,
    MatchingStorage,
    concat_keypoints,
)


def read_image(path: FilePath) -> np.ndarray:
    return cv2.imread(str(path))


class Matcher: ...


class LocalFeatureMatcher(Matcher):
    def __init__(self, refiner: Optional[PANetRefiner] = None):
        self.refiner = refiner

    def __call__(
        self,
        path1: FilePath,
        path2: FilePath,
        shape1: tuple[int, int],  # (H, W)
        shape2: tuple[int, int],  # (H, W)
        feature_storage: LocalFeatureStorage,
        matching_storage: Optional[MatchingStorage] = None,
        cropper: Optional[OverlapRegionCropper] = None,
        image_reader: Optional[Callable] = None,
    ) -> np.ndarray:
        lafs1, kpts1, _, descs1 = feature_storage.get(path1)
        lafs2, kpts2, _, descs2 = feature_storage.get(path2)
        idxs = self.match(descs1, descs2)

        if self.use_overlap_region_cropper and cropper:
            idxs = self.filter_matches_out_of_overlap_region(
                idxs, kpts1, kpts2, cropper
            )

        if self.refiner:
            assert image_reader
            img1 = image_reader(str(path1))
            img2 = image_reader(str(path2))
            new_kpts1, new_kpts2 = self.refiner.refine_matched_keypoints(
                img1, img2, kpts1, kpts2, idxs
            )
            # TODO
            assert isinstance(feature_storage, InMemoryLocalFeatureStorage)
            feature_storage.keypoints[Path(path1).name] = new_kpts1.copy()
            feature_storage.keypoints[Path(path2).name] = new_kpts2.copy()

        if matching_storage:
            if self.min_matches is None or len(idxs) >= self.min_matches:
                matching_storage.add(path1, path2, idxs)
        return idxs

    @property
    def min_matches(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def use_overlap_region_cropper(self) -> bool:
        raise NotImplementedError

    def match(
        self,
        descs1: np.ndarray,
        descs2: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def filter_matches_out_of_overlap_region(
        self,
        idxs: np.ndarray,
        kpts1: np.ndarray,
        kpts2: np.ndarray,
        cropper: OverlapRegionCropper,
    ) -> np.ndarray:
        _, keep1 = cropper.cropper1.remove_keypoints_out_of_bbox(
            kpts1[idxs[:, 0]].copy()
        )
        _, keep2 = cropper.cropper2.remove_keypoints_out_of_bbox(
            kpts2[idxs[:, 1]].copy()
        )
        return idxs[np.intersect1d(keep1, keep2)]


class Line2DFeatureMatcher(Matcher):
    def __init__(self):
        pass

    @property
    def min_matches(self) -> Optional[int]:
        raise NotImplementedError

    def __call__(
        self,
        path1: FilePath,
        path2: FilePath,
        shape1: tuple[int, int],  # (H, W)
        shape2: tuple[int, int],  # (H, W)
        feature_storage: Line2DFeatureStorage,
        matching_storage: Optional[MatchingStorage] = None,
        cropper: Optional[OverlapRegionCropper] = None,
        image_reader: Optional[Callable] = None,
    ) -> np.ndarray:
        segs1, descinfos1 = feature_storage.get(path1)
        segs2, descinfos2 = feature_storage.get(path2)
        idxs = self.match(descinfos1, descinfos2)

        if cropper:
            # TODO
            pass

        if matching_storage:
            if self.min_matches is None or len(idxs) >= self.min_matches:
                matching_storage.add(path1, path2, idxs)
        return idxs

    def match(self, descinfos1: Any, descinfos2: Any) -> np.ndarray:
        raise NotImplementedError


class ComposedLocalFeatureMatcher(LocalFeatureMatcher):
    def __call__(
        self,
        path1: FilePath,
        path2: FilePath,
        shape1: tuple[int, int],  # (H, W)
        shape2: tuple[int, int],  # (H, W)
        matching_storage: Optional[MatchingStorage] = None,
        cropper: Optional[Cropper] = None,
        overlap_region_cropper: Optional[OverlapRegionCropper] = None,
        image_reader: Optional[Callable] = None,
    ) -> np.ndarray:
        raise NotImplementedError


class DetectorFreeMatcher(Matcher):
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
        raise NotImplementedError


class PointTrackingMatcher(Matcher):
    @property
    def extractors(self) -> list[LocalFeatureExtractor]:
        raise NotImplementedError

    def is_hybrid(self) -> bool:
        """
        If True, the matcher registers matched keypoints
        in addition to matchings based on query keypoints

        NOTE
        Required impl_version>=2
        """
        return False

    def __call__(
        self,
        path1: FilePath,
        path2: FilePath,
        keypoint_storage: KeypointStorage,  # in
        matching_storage: MatchingStorage,  # out
        matched_keypoint_storage: MatchedKeypointStorage,  # out
        cropper: Optional[OverlapRegionCropper] = None,
        orientation1: Optional[int] = None,
        orientation2: Optional[int] = None,
        image_reader: Callable | None = None,
    ):
        raise NotImplementedError

    @torch.inference_mode()
    def extract_keypoints(
        self, scene: Scene, progress_bar: tqdm.tqdm | None = None
    ) -> KeypointStorage:
        """NOTE: Matcher should not take `Scene`"""
        storages = []
        for extractor in self.extractors:
            f_storage = InMemoryLocalFeatureStorage()
            extract_all(extractor, scene, storage=f_storage, progress_bar=progress_bar)
            storages.append(f_storage.to_keypoint_storage())

        return concat_keypoints(storages)

    @torch.inference_mode()
    def prepare(
        self,
        image_paths: Sequence[str | Path],
        image_reader: Callable[..., Any] | None = None,
        progress_bar: tqdm.tqdm | None = None,
    ) -> None:
        pass


def run_overlap_region_estimation(
    estimator: OverlapRegionEstimator,
    pairs: list[tuple[int, int]],
    scene: Scene,
    matched_keypoint_storage: MatchedKeypointStorage,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> None:
    for i, (idx1, idx2) in enumerate(pairs):
        path1 = scene.image_paths[idx1]
        path2 = scene.image_paths[idx2]
        shape1 = scene.get_image_shape(path1)
        shape2 = scene.get_image_shape(path2)
        try:
            mkpts1, mkpts2 = matched_keypoint_storage.get(path1, path2)
            bboxes1, bboxes2 = estimator.get_paired_bboxes(
                mkpts1, mkpts2, shape1, shape2
            )
            bbox1 = bboxes1[0]
            bbox2 = bboxes2[0]
            scene.update_overlap_regions(path1, path2, bbox1, bbox2)
        except Exception as e:
            print(f"[run_overlap_region_estimation] Error: {e}")
        if progress_bar:
            progress_bar.set_postfix_str(f"Overlap estimation ({i + 1}/{len(pairs)})")
