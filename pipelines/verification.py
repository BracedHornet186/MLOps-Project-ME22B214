from __future__ import annotations

from collections.abc import Callable
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pycolmap
import pydegensac
import torch
import tqdm

try:
    import poselib
except Exception:
    poselib = None

from scripts.colmap import (
    export_two_view_geometries_from_colmap,
    import_two_view_geometories_into_colmap,
)
from scripts.data import SHOW_MATCHED_KEYPOINT_COUNT, SHOW_STATS, FilePath
from pipelines.config import GeometricVerificationConfig
from pipelines.scene import Scene
from pipelines.visualizer import StorageVisualizer
from postprocesses.config import FSNetConfig, PoseLibRANSACConfig, RANSACConfig
from postprocesses.fsnet import FSNet
from scripts.storage import (
    InMemoryKeypointStorage,
    InMemoryMatchingStorage,
    InMemoryTwoViewGeometryStorage,
    KeypointStorage,
    MatchingStorage,
)


def verify_matches(
    scene: Scene,
    conf: GeometricVerificationConfig,
    keypoint_storage: Optional[KeypointStorage] = None,
    matching_storage: Optional[MatchingStorage] = None,
    id_mappings: Optional[dict[str, int]] = None,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> InMemoryTwoViewGeometryStorage:
    """Verify matched keypoints with geometric verification"""
    database_path = scene.database_path

    if conf.type == "colmap":
        # NOTE
        # Doc: https://github.com/colmap/pycolmap/blob/master/pipeline/match_features.cc
        # Doc: https://github.com/colmap/colmap/blob/dev/src/feature/matching.cc#L604-L614
        # pycolmap.SiftMatchingOptions:
        #    num_threads = -1
        #    gpu_index = -1
        #    max_ratio = 0.8
        #    max_distance = 0.7
        #    cross_check = True
        #    max_num_matches = 32768
        #    max_error = 4.0            (For RANSAC)
        #    confidence = 0.999         (For RANSAC)
        #    min_num_trials = 100       (For RANSAC)
        #    max_num_trials = 10000     (For RANSAC)
        #    min_inlier_ratio = 0.25    (For RANSAC)
        #    min_num_inliers = 15       (For RANSAC)
        #    multiple_models = False
        #    guided_matching = False
        #    planar_scene = False
        if hasattr(pycolmap, "TwoViewGeometryOptions"):
            verification_options = pycolmap.TwoViewGeometryOptions()
            if conf.ransac:
                verification_options.max_error = conf.ransac.threshold
                verification_options.confidence = conf.ransac.confidence
                verification_options.max_num_trials = conf.ransac.max_iters
            pycolmap.match_exhaustive(database_path, verification_options=verification_options)
        else:
            options = pycolmap.SiftMatchingOptions()
            if hasattr(options, "num_threads"):
                options.num_threads = 1
            if conf.ransac:
                options.max_error = conf.ransac.threshold
                options.confidence = conf.ransac.confidence
                options.max_num_trials = conf.ransac.max_iters
            pycolmap.match_exhaustive(database_path, options)
        g_storage = export_two_view_geometries_from_colmap(
            database_path=str(database_path)
        )
    elif conf.type == "custom":
        assert keypoint_storage
        assert matching_storage
        assert id_mappings
        assert conf.ransac
        g_storage = custom_verification(
            keypoint_storage,
            matching_storage,
            conf.ransac,
            database_path=database_path,
            id_mappings=id_mappings,
            progress_bar=progress_bar,
        )
    elif conf.type == "poselib":
        assert keypoint_storage
        assert matching_storage
        assert id_mappings
        assert conf.poselib_ransac
        g_storage = poselib_ransac_verification(
            keypoint_storage,
            matching_storage,
            conf.poselib_ransac,
            database_path=database_path,
            id_mappings=id_mappings,
            progress_bar=progress_bar,
        )
    elif conf.type == "fsnet":
        assert keypoint_storage
        assert matching_storage
        assert id_mappings
        assert conf.fsnet
        # TODO
        device = torch.device("cuda:0")
        g_storage = fsnet_verification(
            keypoint_storage,
            matching_storage,
            scene,
            conf.fsnet,
            device,
            database_path=database_path,
            id_mappings=id_mappings,
            progress_bar=progress_bar,
        )
    else:
        raise RuntimeError
    return g_storage


def custom_verification(
    keypoint_storage: KeypointStorage,
    matching_storage: MatchingStorage,
    conf: RANSACConfig,
    database_path: FilePath,
    id_mappings: dict[str, int],
    progress_bar: Optional[tqdm.tqdm] = None,
) -> InMemoryTwoViewGeometryStorage:
    k_storage: InMemoryKeypointStorage = keypoint_storage.to_memory()
    m_storage: InMemoryMatchingStorage = matching_storage.to_memory()
    g_storage = InMemoryTwoViewGeometryStorage()

    num_pairs = 0
    for key1, group in m_storage:
        for key2 in group.keys():
            num_pairs += 1

    count = 0
    for key1, group in m_storage:
        for key2, idxs in group.items():
            count += 1
            if progress_bar:
                progress_bar.set_postfix_str(f"Verification ({count}/{num_pairs})")

            mkpts1 = k_storage.keypoints[key1][idxs[:, 0]].copy()
            mkpts2 = k_storage.keypoints[key2][idxs[:, 1]].copy()

            try:
                F, inliers = run_ransac(mkpts1, mkpts2, conf)
                # NOTE: inliers: Shape(N, 1), value={0, 1}
            except Exception as e:
                print(f"RANSAC error: {e}")
                continue

            if SHOW_STATS:
                print(f"{key1} {key2}: #matches={len(mkpts1)}, #inliers={sum(inliers)}")

            if len(inliers) == 0:
                continue

            inlier_mask = (inliers > 0).reshape(-1)
            if conf.min_inliers is not None and sum(inlier_mask) < conf.min_inliers:
                continue

            inlier_idxs = idxs[inlier_mask]
            g_storage.add(key1, key2, inlier_idxs, F)

    # if SHOW_STATS:
    #    visualizer = StorageVisualizer(
    #        k_storage, m_storage, geometry_storage=g_storage
    #    )
    #    visualizer.show_stats()

    import_two_view_geometories_into_colmap(
        g_storage, id_mappings, database_path=str(database_path)
    )
    return g_storage


def poselib_ransac_verification(
    keypoint_storage: KeypointStorage,
    matching_storage: MatchingStorage,
    conf: PoseLibRANSACConfig,
    database_path: FilePath,
    id_mappings: dict[str, int],
    progress_bar: Optional[tqdm.tqdm] = None,
) -> InMemoryTwoViewGeometryStorage:
    assert poselib is not None
    ransac_options = conf.to_poselib_ransac_options()
    print(f"[poselib_ransac_verification] poselib: {poselib is not None}")
    print(f"[poselib_ransac_verification] ransac_options: {ransac_options}")

    k_storage: InMemoryKeypointStorage = keypoint_storage.to_memory()
    m_storage: InMemoryMatchingStorage = matching_storage.to_memory()
    g_storage = InMemoryTwoViewGeometryStorage()

    num_pairs = 0
    for key1, group in m_storage:
        for key2 in group.keys():
            num_pairs += 1

    count = 0
    for key1, group in m_storage:
        for key2, idxs in group.items():
            count += 1
            if progress_bar:
                progress_bar.set_postfix_str(f"Verification ({count}/{num_pairs})")

            mkpts1 = k_storage.keypoints[key1][idxs[:, 0]].copy()
            mkpts2 = k_storage.keypoints[key2][idxs[:, 1]].copy()

            F, info = poselib.estimate_fundamental(mkpts1, mkpts2, ransac_options, {})
            inliers = np.array(info["inliers"]).astype(np.int32)
            # NOTE: inliers: Shape(N, 1), value={0, 1}

            if SHOW_STATS:
                print(f"{key1} {key2}: #matches={len(mkpts1)}, #inliers={sum(inliers)}")

            if len(inliers) == 0:
                continue

            inlier_mask = (inliers > 0).reshape(-1)
            if conf.min_inliers is not None and sum(inlier_mask) < conf.min_inliers:
                continue

            inlier_idxs = idxs[inlier_mask]
            g_storage.add(key1, key2, inlier_idxs, F)

    import_two_view_geometories_into_colmap(
        g_storage, id_mappings, database_path=str(database_path)
    )
    return g_storage


def fsnet_verification(
    keypoint_storage: KeypointStorage,
    matching_storage: MatchingStorage,
    scene: Scene,
    conf: FSNetConfig,
    device: torch.device,
    database_path: FilePath,
    id_mappings: dict[str, int],
    progress_bar: Optional[tqdm.tqdm] = None,
) -> InMemoryTwoViewGeometryStorage:
    k_storage: InMemoryKeypointStorage = keypoint_storage.to_memory()
    m_storage: InMemoryMatchingStorage = matching_storage.to_memory()
    g_storage = InMemoryTwoViewGeometryStorage()

    fsnet = FSNet(conf, scene.get_image, device=device)

    num_pairs = 0
    for key1, group in m_storage:
        for key2 in group.keys():
            num_pairs += 1

    count = 0
    for key1, group in m_storage:
        for key2, idxs in group.items():
            count += 1
            if progress_bar:
                progress_bar.set_postfix_str(f"Verification ({count}/{num_pairs})")

            mkpts1 = k_storage.keypoints[key1][idxs[:, 0]].copy()
            mkpts2 = k_storage.keypoints[key2][idxs[:, 1]].copy()

            F_list = []
            inliers_list = []

            for rconf in conf.ransac_list:
                F, inliers = run_ransac(mkpts1, mkpts2, rconf)
                # NOTE: inliers: Shape(N, 1), value={0, 1}
                F_list.append(F)
                inliers_list.append(inliers)

            path1 = scene.image_paths[scene.short_key_to_idx(key1)]
            path2 = scene.image_paths[scene.short_key_to_idx(key2)]
            best_fmat_id = fsnet(str(path1), str(path2), F_list)

            F = F_list[best_fmat_id]
            inliers = inliers_list[best_fmat_id]

            if SHOW_STATS:
                print(
                    f"{key1} {key2}: bestF={best_fmat_id}, #matches={len(mkpts1)}, #inliers={sum(inliers)}"
                )

            if len(inliers) == 0:
                continue

            inlier_mask = (inliers > 0).reshape(-1)
            if conf.min_inliers is not None and sum(inlier_mask) < conf.min_inliers:
                continue

            inlier_idxs = idxs[inlier_mask]
            g_storage.add(key1, key2, inlier_idxs, F)

    # if SHOW_STATS:
    #    visualizer = StorageVisualizer(
    #        k_storage, m_storage, geometry_storage=g_storage
    #    )
    #    visualizer.show_stats()

    import_two_view_geometories_into_colmap(
        g_storage, id_mappings, database_path=str(database_path)
    )
    return g_storage


def compute_ransac_inlier_counts(
    keypoint_storage: KeypointStorage,
    matching_storage: MatchingStorage,
    conf: RANSACConfig,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> dict[str, dict[str, int]]:
    k_storage: InMemoryKeypointStorage = keypoint_storage.to_memory()
    m_storage: InMemoryMatchingStorage = matching_storage.to_memory()

    num_pairs = 0
    inlier_counts = {}
    for key1, group in m_storage:
        if key1 not in inlier_counts:
            inlier_counts[key1] = {}
        for key2 in group.keys():
            num_pairs += 1
            inlier_counts[key1][key2] = 0

    count = 0
    for key1, group in m_storage:
        for key2, idxs in group.items():
            count += 1
            if progress_bar:
                progress_bar.set_postfix_str(f"Verification ({count}/{num_pairs})")

            try:
                mkpts1 = k_storage.keypoints[key1][idxs[:, 0]].copy()
                mkpts2 = k_storage.keypoints[key2][idxs[:, 1]].copy()

                F, inliers = run_ransac(mkpts1, mkpts2, conf)
                # NOTE: inliers: Shape(N, 1), value={0, 1}

                if len(inliers) == 0:
                    continue

                inlier_mask = (inliers > 0).reshape(-1)
                inlier_counts[key1][key2] = int(sum(inlier_mask))
            except Exception as e:
                print(f"[compute_ransac_inlier_counts] Error: {e}")

    return inlier_counts


def run_ransac(
    matched_kpts0: np.ndarray,
    matched_kpts1: np.ndarray,
    conf: RANSACConfig,
    min_matches_required: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    if len(matched_kpts0) < min_matches_required:
        F = np.zeros((3, 3))
        inliers = np.empty((0, 2))
        return F, inliers

    if conf.method in ("magsac",):
        method = cv2.USAC_MAGSAC
        F, inliers = cv2.findFundamentalMat(
            matched_kpts0,
            matched_kpts1,
            method,
            conf.threshold,
            conf.confidence,
            conf.max_iters,
        )
    elif conf.method == "gc-ransac":
        F, inliers = cv2.findFundamentalMat(
            matched_kpts0,
            matched_kpts1,
            cv2.USAC_ACCURATE,
            conf.threshold,
            conf.confidence,
            conf.max_iters,
        )
    elif conf.method == "degensac":
        F, inliers = pydegensac.findFundamentalMatrix(
            matched_kpts0,
            matched_kpts1,
            px_th=conf.threshold,
            conf=conf.confidence,
            max_iters=conf.max_iters,
        )
    else:
        raise ValueError
    return F, inliers
