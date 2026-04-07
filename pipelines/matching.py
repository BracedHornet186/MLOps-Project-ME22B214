from __future__ import annotations

import copy
import warnings
from typing import Literal, Optional

import torch
import tqdm

from extractor import LocalFeatureExtractor, extract_all
from features.factory import create_local_feature_handler
from matchers.base import (
    DetectorFreeMatcher,
    Line2DFeatureMatcher,
    LocalFeatureMatcher,
    PointTrackingMatcher,
)
from matchers.factory import create_detector_free_matcher, create_local_feature_matcher
from pipelines.common import Scene
from pipelines.config import PreMatchingConfig
from postprocesses.coarse_match import run_coarse_matching_postprocess
from postprocesses.config import MatchingFilterConfig
from postprocesses.matching_filter import create_matching_filter
from preprocesses.config import SegmentationConfig
from preprocesses.region import OverlapRegionCropper
from storage import (
    InMemoryKeypointStorage,
    InMemoryLocalFeatureStorage,
    InMemoryMatchedKeypointStorage,
    InMemoryMatchingStorage,
    KeypointStorage,
    Line2DFeatureStorage,
    Line2DSegmentStorage,
    LocalFeatureStorage,
    MatchedKeypointStorage,
    MatchingStorage,
    filter_local_features_by_segmentation_mask,
    filter_matched_keypoints_by_mask_regions,
    fuse_matching_sets_late,
)


def run_local_feature_matching(
    matcher: LocalFeatureMatcher,
    pairs: list[tuple[int, int]],
    scene: Scene,
    feature_storage: LocalFeatureStorage,
    matching_storage: Optional[MatchingStorage] = None,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> None:
    for i, (idx1, idx2) in enumerate(pairs):
        path1 = scene.image_paths[idx1]
        path2 = scene.image_paths[idx2]
        shape1 = scene.get_image_shape(path1)
        shape2 = scene.get_image_shape(path2)
        cropper = scene.create_overlap_region_cropper(
            path1, path2, enable_scale_alignment=False
        )

        matcher(
            path1,
            path2,
            shape1,
            shape2,
            feature_storage,
            matching_storage=matching_storage,
            cropper=cropper,
            image_reader=scene.get_image,
        )
        if progress_bar:
            progress_bar.set_postfix_str(
                f"Local feature matching ({i + 1}/{len(pairs)})"
            )


def run_line2d_feature_matching(
    matcher: Line2DFeatureMatcher,
    pairs: list[tuple[int, int]],
    scene: Scene,
    feature_storage: Line2DFeatureStorage,
    matching_storage: Optional[MatchingStorage] = None,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> None:
    for i, (idx1, idx2) in enumerate(pairs):
        path1 = scene.image_paths[idx1]
        path2 = scene.image_paths[idx2]
        shape1 = scene.get_image_shape(path1)
        shape2 = scene.get_image_shape(path2)
        cropper = scene.create_overlap_region_cropper(
            path1, path2, enable_scale_alignment=False
        )

        matcher(
            path1,
            path2,
            shape1,
            shape2,
            feature_storage,
            matching_storage=matching_storage,
            cropper=cropper,
            image_reader=scene.get_image,
        )
        if progress_bar:
            progress_bar.set_postfix_str(
                f"Line2d feature matching ({i + 1}/{len(pairs)})"
            )


def run_detector_free_matching(
    matcher: DetectorFreeMatcher,
    pairs: list[tuple[int, int]],
    scene: Scene,
    matched_keypoint_storage: MatchedKeypointStorage,
    keypoint_storage: Optional[KeypointStorage] = None,  # For output
    matching_storage: Optional[MatchingStorage] = None,  # For output
    apply_round: bool = True,
    mkpts_decoupling_method: Literal["imc2023", "detector_free_sfm"] = "imc2023",
    cropper_type: Literal["overlap", "roi", "overlap-or-roi", "ignore"] = "overlap",
    enable_cropper_scaling: bool = True,
    matching_filter_conf: Optional[MatchingFilterConfig] = None,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> None:
    for i, (idx1, idx2) in enumerate(pairs):
        path1 = scene.image_paths[idx1]
        path2 = scene.image_paths[idx2]

        cropper = scene.create_overlap_region_cropper(
            path1,
            path2,
            enable_scale_alignment=enable_cropper_scaling,
            cropper_type=cropper_type,
        )

        orientation1 = scene.get_orientation_degree(path1)
        orientation2 = scene.get_orientation_degree(path2)

        matcher(
            path1,
            path2,
            matched_keypoint_storage,
            cropper=cropper,
            orientation1=orientation1,
            orientation2=orientation2,
            image_reader=scene.get_image,
        )

        if matched_keypoint_storage and scene.mask_bboxes:
            print("Filtering matched keypoints based on mask regions")
            assert isinstance(matched_keypoint_storage, InMemoryMatchedKeypointStorage)
            matched_keypoint_storage = filter_matched_keypoints_by_mask_regions(
                matched_keypoint_storage, scene
            )

        if progress_bar:
            progress_bar.set_postfix_str(
                f"Detector free matching ({i + 1}/{len(pairs)})"
            )

    if keypoint_storage is not None and matching_storage is not None:
        if mkpts_decoupling_method == "imc2023":
            keypoint_storage, matching_storage = (
                matched_keypoint_storage.to_keypoints_and_matches(
                    keypoint_storage=keypoint_storage,
                    matching_storage=matching_storage,
                    apply_round=apply_round,
                )
            )
            if matching_filter_conf:
                assert isinstance(keypoint_storage, InMemoryKeypointStorage)
                assert isinstance(matching_storage, InMemoryMatchingStorage)
                matching_filter = create_matching_filter(matching_filter_conf)
                matching_filter.run(
                    keypoint_storage, matching_storage, scene, progress_bar=progress_bar
                )
        elif mkpts_decoupling_method == "detector_free_sfm":
            print(
                "Running coarse matching postprocess to decouple matched keypoints ..."
            )
            match_round_ratio = None
            keypoint_storage, matching_storage = run_coarse_matching_postprocess(
                matched_keypoint_storage,
                scene,
                match_round_ratio=match_round_ratio,
                pair_name_split=" ",
                keypoint_storage=keypoint_storage,  # type: ignore
                matching_storage=matching_storage,  # type: ignore
            )
            if matching_filter_conf:
                assert isinstance(keypoint_storage, InMemoryKeypointStorage)
                assert isinstance(matching_storage, InMemoryMatchingStorage)
                matching_filter = create_matching_filter(matching_filter_conf)
                matching_filter.run(
                    keypoint_storage, matching_storage, scene, progress_bar=progress_bar
                )
        else:
            raise NotImplementedError


def run_pre_matching_with_detector_free_matching(
    matcher: DetectorFreeMatcher,
    pairs: list[tuple[int, int]],
    scene: Scene,
    matched_keypoint_storage: Optional[MatchedKeypointStorage] = None,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> MatchedKeypointStorage:
    if matched_keypoint_storage is None:
        matched_keypoint_storage = InMemoryMatchedKeypointStorage()

    for i, (idx1, idx2) in enumerate(pairs):
        path1 = scene.image_paths[idx1]
        path2 = scene.image_paths[idx2]
        if matched_keypoint_storage.has(path1, path2):
            continue

        orientation1 = scene.get_orientation_degree(path1)
        orientation2 = scene.get_orientation_degree(path2)

        matcher(
            path1,
            path2,
            matched_keypoint_storage,
            orientation1=orientation1,
            orientation2=orientation2,
            image_reader=scene.get_image,
        )
        if progress_bar:
            progress_bar.set_postfix_str(f"Pre-Matching ({i + 1}/{len(pairs)})")

    return matched_keypoint_storage


def run_pre_matching_with_local_feature_matching(
    extractor: LocalFeatureExtractor,
    matcher: LocalFeatureMatcher,
    pairs: list[tuple[int, int]],
    scene: Scene,
    line2d_seg_storage: Optional[Line2DSegmentStorage] = None,
    segmentation_conf: Optional[SegmentationConfig] = None,
    filter_by_segmentation_if_provided: bool = False,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> MatchedKeypointStorage:
    m_storage = InMemoryMatchingStorage()
    f_storage = InMemoryLocalFeatureStorage()

    # Extract local features from all images in the scene
    extract_all(
        extractor,
        scene,
        line2d_seg_storage=line2d_seg_storage,
        storage=f_storage,
        progress_bar=progress_bar,
    )

    if segmentation_conf and filter_by_segmentation_if_provided:
        f_storage = filter_local_features_by_segmentation_mask(f_storage, scene)
        f_storage = f_storage.to_memory()

    # Match all pairs
    run_local_feature_matching(
        matcher,
        pairs,
        scene,
        feature_storage=f_storage,
        matching_storage=m_storage,
        progress_bar=progress_bar,
    )

    k_storage = f_storage.to_keypoint_storage()
    return InMemoryMatchedKeypointStorage().from_keypoints_and_matches(
        k_storage, m_storage
    )


def run_pre_matching(
    conf: PreMatchingConfig,
    pairs: list[tuple[int, int]],
    scene: Scene,
    line2d_seg_storage: Optional[Line2DSegmentStorage] = None,
    segmentation_conf: Optional[SegmentationConfig] = None,
    device: Optional[torch.device] = None,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> list[MatchedKeypointStorage]:
    prematch_mk_storage_list = []
    for c in conf.matchers:
        print(f"[run_pre_matching] {str(c)}")
        if c.type == "local_feature":
            assert c.local_feature
            assert c.local_feature_matcher
            matcher = create_local_feature_matcher(
                c.local_feature_matcher, device=device
            )
            handler = create_local_feature_handler(c.local_feature, device=device)
            extractor = LocalFeatureExtractor(c.local_feature, handler)

            mk_storage = run_pre_matching_with_local_feature_matching(
                extractor,
                matcher,
                pairs,
                scene,
                line2d_seg_storage=line2d_seg_storage,
                segmentation_conf=segmentation_conf,
                filter_by_segmentation_if_provided=conf.filter_by_segmentation_if_provided,
                progress_bar=progress_bar,
            )
            prematch_mk_storage_list.append(mk_storage)
        elif c.type == "detector_free":
            assert c.detector_free_matcher
            matcher = create_detector_free_matcher(
                c.detector_free_matcher, device=device
            )

            mk_storage = run_pre_matching_with_detector_free_matching(
                matcher, pairs, scene, progress_bar=progress_bar
            )
            prematch_mk_storage_list.append(mk_storage)
        else:
            raise ValueError(c.type)

    return prematch_mk_storage_list


def run_point_tracking_matching(
    matcher: PointTrackingMatcher,
    pairs: list[tuple[int, int]],
    scene: Scene,
    keypoint_storage: KeypointStorage,  # For output
    matching_storage: MatchingStorage,  # For output
    matched_keypoint_storage: MatchedKeypointStorage,  # For output
    impl_version: Literal["v1", "v2"] = "v1",
    apply_round: bool = True,
    mkpts_decoupling_method: Literal["imc2023", "detector_free_sfm"] = "imc2023",
    matching_filter_conf: Optional[MatchingFilterConfig] = None,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> None:
    if impl_version == "v1":
        print(
            "run_point_tracking_matching: Set 'v2' to `impl_version` !!!!!!! "
            "This function is running with incorrect settings !!"
        )
    query_point_storage = matcher.extract_keypoints(scene, progress_bar=progress_bar)
    matcher.prepare(
        scene.image_paths, image_reader=scene.get_image, progress_bar=progress_bar
    )

    for i, (idx1, idx2) in enumerate(pairs):
        path1 = scene.image_paths[idx1]
        path2 = scene.image_paths[idx2]

        # NOTE: PointTrackingMatcher-based matching does not support cropper and orientation normalizer

        matcher(
            path1,
            path2,
            query_point_storage,
            matching_storage,
            matched_keypoint_storage,
            image_reader=scene.get_image,
        )

        if matched_keypoint_storage and scene.mask_bboxes:
            print("Filtering matched keypoints based on mask regions")
            assert isinstance(matched_keypoint_storage, InMemoryMatchedKeypointStorage)
            matched_keypoint_storage = filter_matched_keypoints_by_mask_regions(
                matched_keypoint_storage, scene
            )

        if progress_bar:
            progress_bar.set_postfix_str(
                f"Point tracking matching ({i + 1}/{len(pairs)})"
            )

    if impl_version == "v2":
        assert isinstance(keypoint_storage, InMemoryKeypointStorage)
        assert isinstance(matching_storage, InMemoryMatchingStorage)
        assert isinstance(query_point_storage, InMemoryKeypointStorage)

        if not matcher.is_hybrid():
            # NOTE
            #  - Import keypoints from query points
            #  - Matching keypoint storage has been updated in matcher()
            keypoint_storage.keypoints = query_point_storage.keypoints.copy()
            return

        # Hybrid matcher return matched keypoints additionally
        if mkpts_decoupling_method == "imc2023":
            _keypoint_storage, _matching_storage = (
                matched_keypoint_storage.to_keypoints_and_matches(
                    apply_round=apply_round,
                )
            )
            assert isinstance(_keypoint_storage, InMemoryKeypointStorage)
            assert isinstance(_matching_storage, InMemoryMatchingStorage)
            if matching_filter_conf:
                assert isinstance(keypoint_storage, InMemoryKeypointStorage)
                assert isinstance(matching_storage, InMemoryMatchingStorage)
                matching_filter = create_matching_filter(matching_filter_conf)
                matching_filter.run(
                    _keypoint_storage,
                    _matching_storage,
                    scene,
                    progress_bar=progress_bar,
                )

            concat_k_storage, concat_m_storage = fuse_matching_sets_late(
                [
                    (query_point_storage, matching_storage),
                    (_keypoint_storage, _matching_storage),
                ],
                scene,
            )
            keypoint_storage.keypoints = concat_k_storage.keypoints.copy()
            matching_storage.matches = copy.deepcopy(concat_m_storage.matches)
        else:
            raise NotImplementedError
    elif mkpts_decoupling_method == "imc2023":
        keypoint_storage, matching_storage = (
            matched_keypoint_storage.to_keypoints_and_matches(
                keypoint_storage=keypoint_storage,
                matching_storage=matching_storage,
                apply_round=apply_round,
            )
        )
        if matching_filter_conf:
            assert isinstance(keypoint_storage, InMemoryKeypointStorage)
            assert isinstance(matching_storage, InMemoryMatchingStorage)
            matching_filter = create_matching_filter(matching_filter_conf)
            matching_filter.run(
                keypoint_storage, matching_storage, scene, progress_bar=progress_bar
            )
    elif mkpts_decoupling_method == "detector_free_sfm":
        print("Running coarse matching postprocess to decouple matched keypoints ...")
        match_round_ratio = None
        keypoint_storage, matching_storage = run_coarse_matching_postprocess(
            matched_keypoint_storage,
            scene,
            match_round_ratio=match_round_ratio,
            pair_name_split=" ",
            keypoint_storage=keypoint_storage,  # type: ignore
            matching_storage=matching_storage,  # type: ignore
        )
        if matching_filter_conf:
            assert isinstance(keypoint_storage, InMemoryKeypointStorage)
            assert isinstance(matching_storage, InMemoryMatchingStorage)
            matching_filter = create_matching_filter(matching_filter_conf)
            matching_filter.run(
                keypoint_storage, matching_storage, scene, progress_bar=progress_bar
            )
    else:
        raise NotImplementedError
