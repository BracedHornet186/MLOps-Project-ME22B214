from __future__ import annotations

import gc
from typing import Any, Optional

import hloc.match_dense
import hloc.utils.io
import numpy as np
import tqdm

from scripts.extractor import (
    Line2DFeatureExtractor,
    LocalFeatureExtractor,
    extract_all,
    extract_line2d_features_all,
)
from matchers.base import (
    DetectorFreeMatcher,
    Line2DFeatureMatcher,
    LocalFeatureMatcher,
    PointTrackingMatcher,
)
from matchers.config import DetectorFreeMatcherConfig, PointTrackingMatcherConfig
from pipelines.config import HLocMatchDenseConfig
from pipelines.matching import (
    run_detector_free_matching,
    run_line2d_feature_matching,
    run_local_feature_matching,
    run_point_tracking_matching,
)
from pipelines.scene import Scene
from preprocesses.config import SegmentationConfig
from scripts.storage import (
    InMemoryKeypointStorage,
    InMemoryLine2DFeatureStorage,
    InMemoryLine2DSegmentStorage,
    InMemoryLocalFeatureStorage,
    InMemoryMatchedKeypointStorage,
    InMemoryMatchingStorage,
    KeypointStorage,
    Line2DFeatureStorage,
    Line2DSegmentStorage,
    MatchedKeypointStorage,
    concat_keypoints,
    concat_matched_keypoints,
    filter_local_features_by_segmentation_mask,
    show_keypoint_matching_coverage,
)
from utils.hloc_io import (
    export_keypoints_to_hloc,
    export_matched_keypoints_to_hloc,
    import_keypoints_from_hloc,
    import_matched_keypoints_from_hloc,
    import_matching_from_hloc,
)


def run_local_feature_extraction_and_matching_stage(
    scene: Scene,
    pairs: list[tuple[int, int]],
    local_feature_extractors: list[LocalFeatureExtractor],
    local_feature_matchers: list[LocalFeatureMatcher],
    line2d_seg_storage: Optional[Line2DSegmentStorage] = None,
    segmentation_conf: Optional[SegmentationConfig] = None,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> tuple[list[InMemoryKeypointStorage], list[InMemoryMatchingStorage]]:
    """local feature extraction -> local feature matching"""
    local_feature_keypoint_storages: list[InMemoryKeypointStorage] = []
    local_feature_matching_storages: list[InMemoryMatchingStorage] = []
    for extractor, matcher in zip(local_feature_extractors, local_feature_matchers):
        m_storage = InMemoryMatchingStorage()
        f_storage = InMemoryLocalFeatureStorage()

        gc.collect()

        # Extract local features from all images in the scene
        extract_all(
            extractor,
            scene,
            line2d_seg_storage=line2d_seg_storage,
            storage=f_storage,
            progress_bar=progress_bar,
        )

        if segmentation_conf:
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
        k_storage = f_storage.to_memory().to_keypoint_storage()

        local_feature_keypoint_storages.append(k_storage)
        local_feature_matching_storages.append(m_storage)

    return local_feature_keypoint_storages, local_feature_matching_storages


def run_line2d_feature_extraction_and_matching_stage(
    scene: Scene,
    pairs: list[tuple[int, int]],
    line2d_feature_extractors: list[Line2DFeatureExtractor],
    line2d_feature_matchers: list[Line2DFeatureMatcher],
    progress_bar: Optional[tqdm.tqdm] = None,
) -> tuple[list[InMemoryLine2DSegmentStorage], list[InMemoryMatchingStorage]]:
    """line2d feature extraction -> line2d feature matching"""
    line2d_segment_storages: list[InMemoryLine2DSegmentStorage] = []
    line2d_matching_storages: list[InMemoryMatchingStorage] = []
    for extractor, matcher in zip(line2d_feature_extractors, line2d_feature_matchers):
        m_storage = InMemoryMatchingStorage()
        f_storage = InMemoryLine2DFeatureStorage()

        gc.collect()

        # Extract local features from all images in the scene
        extract_line2d_features_all(
            extractor, scene, storage=f_storage, progress_bar=progress_bar
        )

        # Match all pairs
        # TODO
        run_line2d_feature_matching(
            matcher,
            pairs,
            scene,
            feature_storage=f_storage,
            matching_storage=m_storage,
            progress_bar=progress_bar,
        )
        seg_storage = InMemoryLine2DSegmentStorage().from_line2d_feature_storage(
            f_storage
        )

        line2d_segment_storages.append(seg_storage)
        line2d_matching_storages.append(m_storage)

    return line2d_segment_storages, line2d_matching_storages


def run_detector_free_matching_stage(
    scene: Scene,
    pairs: list[tuple[int, int]],
    detector_free_matchers: list[DetectorFreeMatcher],
    detector_free_matcher_configs: list[DetectorFreeMatcherConfig],
    progress_bar: Optional[tqdm.tqdm] = None,
) -> tuple[list[InMemoryKeypointStorage], list[InMemoryMatchingStorage]]:
    detector_free_keypoint_storages: list[InMemoryKeypointStorage] = []
    detector_free_matching_storages: list[InMemoryMatchingStorage] = []
    for m_conf, matcher in zip(detector_free_matcher_configs, detector_free_matchers):
        mk_storage = InMemoryMatchedKeypointStorage()
        k_storage = InMemoryKeypointStorage()
        m_storage = InMemoryMatchingStorage()

        # Match all pairs
        run_detector_free_matching(
            matcher,
            pairs,
            scene,
            matched_keypoint_storage=mk_storage,
            keypoint_storage=k_storage,
            matching_storage=m_storage,
            apply_round=m_conf.apply_round,
            mkpts_decoupling_method=m_conf.mkpts_decoupling_method,
            cropper_type=m_conf.cropper_type,
            matching_filter_conf=m_conf.matching_filter,
            progress_bar=progress_bar,
        )

        detector_free_keypoint_storages.append(k_storage)
        detector_free_matching_storages.append(m_storage)

    return detector_free_keypoint_storages, detector_free_matching_storages


def run_detector_free_matching_stage_with_hloc_match_dense(
    scene: Scene,
    pairs: list[tuple[int, int]],
    detector_free_matchers: list[DetectorFreeMatcher],
    detector_free_matcher_configs: list[DetectorFreeMatcherConfig],
    hloc_conf: HLocMatchDenseConfig,
    local_feature_keypoint_storages: Optional[list[KeypointStorage]] = None,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> tuple[list[InMemoryKeypointStorage], list[InMemoryMatchingStorage]]:
    storages: list[MatchedKeypointStorage] = []
    for m_conf, matcher in zip(detector_free_matcher_configs, detector_free_matchers):
        mk_storage = InMemoryMatchedKeypointStorage()

        # Match all pairs
        run_detector_free_matching(
            matcher,
            pairs,
            scene,
            matched_keypoint_storage=mk_storage,
            keypoint_storage=None,
            matching_storage=None,
            apply_round=m_conf.apply_round,
            mkpts_decoupling_method=m_conf.mkpts_decoupling_method,
            cropper_type=m_conf.cropper_type,
            matching_filter_conf=m_conf.matching_filter,
            progress_bar=progress_bar,
        )
        storages.append(mk_storage)

    name_pairs = [scene.get_paired_names(pair) for pair in pairs]
    storage = concat_matched_keypoints(storages)
    storage.fill_empty_matches(name_pairs)

    hloc_match_path = scene.hloc_dir / "dense_match.h5"
    hloc_feature_path = scene.hloc_dir / "dense_match_feature.h5"

    export_matched_keypoints_to_hloc(storage, hloc_match_path)

    name_pairs = hloc.match_dense.find_unique_new_pairs(name_pairs)

    if hloc_conf.use_local_features:
        assert local_feature_keypoint_storages is not None

        local_feature_keypoint_storage = concat_keypoints(
            local_feature_keypoint_storages
        )
        export_keypoints_to_hloc(local_feature_keypoint_storage, hloc_feature_path)

        feature_paths_refs = [hloc_feature_path]

        cpdict, bindict = hloc.match_dense.load_keypoints(
            {"max_error": hloc_conf.max_error, "cell_size": hloc_conf.cell_size},
            feature_paths_refs,
            quantize=set(sum(name_pairs, ())),
        )

        cpdict = hloc.match_dense.aggregate_matches(
            conf={"max_error": hloc_conf.max_error, "cell_size": hloc_conf.cell_size},
            pairs=name_pairs,
            match_path=hloc_match_path,
            feature_path=hloc_feature_path,
            required_queries=set(sum(name_pairs, ())),
            max_kps=hloc_conf.max_keypoints,
            cpdict=cpdict,
            bindict=bindict,
        )
    else:
        cpdict = hloc.match_dense.aggregate_matches(
            conf={"max_error": hloc_conf.max_error, "cell_size": hloc_conf.cell_size},
            pairs=name_pairs,
            match_path=hloc_match_path,
            feature_path=hloc_feature_path,
            required_queries=set(sum(name_pairs, ())),
            max_kps=hloc_conf.max_keypoints,
        )

    if hloc_conf.max_keypoints is not None:
        hloc.match_dense.assign_matches(
            pairs=name_pairs,
            match_path=hloc_match_path,
            keypoints=cpdict,
            max_error=hloc_conf.max_error,
        )

    detector_free_keypoint_storage = import_keypoints_from_hloc(hloc_feature_path)
    detector_free_matching_storage = import_matching_from_hloc(hloc_match_path)

    return [detector_free_keypoint_storage], [detector_free_matching_storage]


def run_point_tracking_matching_stage(
    scene: Scene,
    pairs: list[tuple[int, int]],
    point_tracking_matchers: list[PointTrackingMatcher],
    point_tracking_matcher_configs: list[PointTrackingMatcherConfig],
    progress_bar: Optional[tqdm.tqdm] = None,
) -> tuple[list[InMemoryKeypointStorage], list[InMemoryMatchingStorage]]:
    keypoint_storages: list[InMemoryKeypointStorage] = []
    matching_storages: list[InMemoryMatchingStorage] = []
    for m_conf, matcher in zip(point_tracking_matcher_configs, point_tracking_matchers):
        mk_storage = InMemoryMatchedKeypointStorage()
        k_storage = InMemoryKeypointStorage()
        m_storage = InMemoryMatchingStorage()

        # Match all pairs
        run_point_tracking_matching(
            matcher,
            pairs,
            scene,
            k_storage,
            m_storage,
            mk_storage,
            impl_version=m_conf.impl_version,
            apply_round=m_conf.apply_round,
            mkpts_decoupling_method="imc2023",
            matching_filter_conf=m_conf.matching_filter,
            progress_bar=progress_bar,
        )

        keypoint_storages.append(k_storage)
        matching_storages.append(m_storage)

    return keypoint_storages, matching_storages
