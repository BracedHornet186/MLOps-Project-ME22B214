from __future__ import annotations

import copy
import subprocess
from pathlib import Path
from typing import Optional, cast

import numpy as np
import pandas as pd
import pycolmap
import torch
import tqdm

from clusterings.factory import MASt3RFPSClustering, create_clustering
from colmap import (
    get_image_id_of_scene_graph_center,
    get_outlier_reconstructions,
    import_into_colmap,
)
from data import DEFAULT_OUTLIER_SCENE_NAME, SAVE_CAMERA_DEBUG_INFO, set_random_seed
from data_schema import DataSchema
from distributed import DistConfig
from matchers.base import run_overlap_region_estimation
from matchers.factory import create_point_tracking_matcher
from matchers.mast3r import MASt3RMatcher
from matchers.mast3r_c2f import MASt3RC2FMatcher
from matchers.mast3r_hybrid import MASt3RHybridMatcher
from pipelines.base import Pipeline
from pipelines.common import (
    Scene,
    create_data_dict,
    init_result_dict_with_scene_clustering,
    iterate_scenes,
    results_to_submission_df,
)
from pipelines.config import IMC2025MASt3RPipelineConfig
from pipelines.matching import run_point_tracking_matching
from pipelines.snapshot import SceneSnapshot
from pipelines.verification import verify_matches
from preprocesses.region import OverlapRegionEstimator
from shortlists.factory import create_shortlist_generator
from storage import (
    InMemoryKeypointStorage,
    InMemoryMatchedKeypointStorage,
    InMemoryMatchingStorage,
)
from utils.camvis import save_camera_debug_info
from utils.imc25.metric import register_by_Horn
from workspace import log


class IMC2025MASt3RPipeline(Pipeline):
    def __init__(
        self,
        conf: IMC2025MASt3RPipelineConfig,
        dist_conf: Optional[DistConfig] = None,
        device: Optional[torch.device] = None,
    ):
        set_random_seed(seed=conf.seed)
        dist_conf = dist_conf or DistConfig.single()
        device = device or torch.device("cpu")

        self.dist_conf = dist_conf
        self.device = device
        self.conf = conf

        # Clustering
        clustering = create_clustering(conf.clustering, device=device)
        assert isinstance(clustering, MASt3RFPSClustering)
        self.clustering: MASt3RFPSClustering = clustering

        # Shortlist
        self.shortlist_generator_in_clustering = None
        if self.conf.shortlist_generator_in_clustering:
            self.shortlist_generator_in_clustering = create_shortlist_generator(
                self.conf.shortlist_generator_in_clustering, device=device
            )
        self.shortlist_generator = create_shortlist_generator(
            self.conf.shortlist_generator, device=device
        )

        # Matchers
        self.matcher = MASt3RMatcher(conf.matcher, device=device)
        self.c2f_matcher = None
        if conf.matcher_c2f:
            self.c2f_matcher = MASt3RC2FMatcher(conf.matcher_c2f, device=device)
        self.hybrid_matcher = None
        if conf.matcher_hybrid:
            self.hybrid_matcher = create_point_tracking_matcher(
                conf.matcher_hybrid, device=self.device
            )

        # Preprocessors
        self.overlap_region_estimator = None
        if conf.overlap_region_estimation:
            self.overlap_region_estimator = OverlapRegionEstimator(
                conf.overlap_region_estimation
            )

    def run(
        self, df: pd.DataFrame, data_schema: DataSchema, save_snapshot: bool = False
    ) -> pd.DataFrame:
        log("IMC2025MASt3RPipeline starts")

        data_dict = create_data_dict(data_schema, df=df, ignore_gt_scene_label=True)
        # results, num_scenes = init_result_dict(data_dict)
        results = init_result_dict_with_scene_clustering(data_dict)
        log(f"The data list has been loaded. # of datasets: {len(results)}")

        iterator = iterate_scenes(data_dict, data_schema)
        progress_bar = tqdm.tqdm(
            total=len(results),
            desc="IMC2025MASt3RPipeline",
            disable=self.dist_conf.is_slave(),
        )

        # NOTE
        # Iterate run_scene() over "datasets" because "scenes" means "datasets" in IMC2025
        seen_datasets = set()
        for scene in iterator:
            if seen_datasets and scene.dataset not in seen_datasets:
                progress_bar.update(1)
            seen_datasets.add(scene.dataset)
            progress_bar.set_description(
                f"IMC2025MASt3RPipeline::{scene.dataset} ({len(seen_datasets)}/{len(results)})"
            )

            assert isinstance(scene, Scene)
            with scene.create_space() as scene:
                outputs = self.run_scene(
                    scene, progress_bar, save_snapshot=save_snapshot
                )
                results[scene.dataset] = outputs
        progress_bar.update(1)

        df = results_to_submission_df(results, schema="imc2025")
        return df

    def run_scene(
        self, scene: Scene, iterator: tqdm.tqdm, save_snapshot: bool = False
    ) -> dict:
        scene.cache_all_images()

        pairs_for_clustering = None
        if self.shortlist_generator_in_clustering:
            pairs_for_clustering = self.shortlist_generator_in_clustering(
                scene, progress_bar=iterator
            )

        clustering_result = self.clustering.run(
            scene.image_paths,
            image_reader=scene.get_image,
            pre_computed_pairs=pairs_for_clustering,
            neighbor_metric=self.conf.clustering_neighbor_metric,
        )
        pre_mkpt_storage = cast(
            InMemoryMatchedKeypointStorage,
            clustering_result.get_output("matched_keypoint_storage"),
        )
        pairwise_scores = cast(
            np.ndarray, clustering_result.get_output("pairwise_score")
        )

        clustered_scenes = clustering_result.to_scene_list(
            scene.dataset, scene.data_schema
        )
        num_clusters = len(clustered_scenes)
        clustered_scene_results = []
        for cluster_idx, clustered_scene in enumerate(clustered_scenes):
            scene.make_output_dir_for_child_scene(clustered_scene)
            mkpt_storage = pre_mkpt_storage.clone_subset(clustered_scene.image_paths)

            pairwise_scores_in_cluster = focus_on_cluster(
                clustered_scene, pairwise_scores
            )
            pairs = self.make_pairs(
                clustered_scene,
                pairwise_scores_in_cluster,
                basic_pair_topk=None,
                iterator=iterator,
            )
            log(
                f"[{clustered_scene.scene}; {cluster_idx + 1}/{num_clusters}] "
                f"# of pairs: {len(pairs)}"
            )

            if self.overlap_region_estimator:
                run_overlap_region_estimation(
                    self.overlap_region_estimator,
                    pairs,
                    clustered_scene,
                    matched_keypoint_storage=mkpt_storage,
                    progress_bar=iterator,
                )
                clustered_scene.make_roi_from_overlap_regions()

            if self.conf.matching_stage_mode == "complementary":
                # Mode: "complementary"
                # ---------------------
                # 1. If a pair has matched keypoints from the clustering stage, re-use the results
                # 2. Otherwise, MASt3RMatcher computes the matching between idx1 and idx2 images
                # Required matchers:
                #   - MASt3RMatcher
                for i, (idx1, idx2) in enumerate(pairs):
                    iterator.set_postfix_str(
                        f"[{clustered_scene.scene}; {cluster_idx + 1}/{num_clusters}] "
                        f"MASt3R matching ({i + 1}/{len(pairs)})"
                    )
                    path1 = clustered_scene.image_paths[idx1]
                    path2 = clustered_scene.image_paths[idx2]
                    if mkpt_storage.has(path1, path2):
                        continue
                    cropper = clustered_scene.create_overlap_region_cropper(
                        path1, path2, cropper_type=self.conf.cropper_type
                    )
                    self.matcher(
                        path1,
                        path2,
                        mkpt_storage,
                        cropper=cropper,
                        image_reader=scene.get_image,
                    )

                keypoint_storage = InMemoryKeypointStorage()
                matching_storage = InMemoryMatchingStorage()
                mkpt_storage.to_keypoints_and_matches(
                    keypoint_storage=keypoint_storage,
                    matching_storage=matching_storage,
                    apply_round=self.conf.round_matched_keypoints,
                )
            elif self.conf.matching_stage_mode == "c2f_override":
                # Mode: "c2f_override"
                # -------------------------------
                # 1. Matched keypoints from the clustering stage will not be used
                # 2. A pair that has matched keypoints from the clustering stage
                #      -> MASt3RC2FMatcher
                # 3. A pair that does not have matched keypoints from the clustering stage
                #      -> MASt3RMatcher
                # Required matchers:
                #   - MASt3RMatcher
                #   - MASt3RC2FMatcher
                for i, (idx1, idx2) in enumerate(pairs):
                    iterator.set_postfix_str(
                        f"[{clustered_scene.scene}; {cluster_idx + 1}/{num_clusters}] "
                        f"MASt3R matching ({i + 1}/{len(pairs)})"
                    )
                    path1 = clustered_scene.image_paths[idx1]
                    path2 = clustered_scene.image_paths[idx2]

                    H1, W1 = scene.image_shapes[str(path1)]
                    H2, W2 = scene.image_shapes[str(path2)]
                    enable_c2f = 512 < max(H1, W1) or 512 < max(H2, W2)
                    if mkpt_storage.has(path1, path2) and enable_c2f:
                        assert self.c2f_matcher is not None
                        self.c2f_matcher(
                            path1,
                            path2,
                            mkpt_storage,
                            image_reader=scene.get_image,
                        )
                    else:
                        cropper = clustered_scene.create_overlap_region_cropper(
                            path1, path2, cropper_type=self.conf.cropper_type
                        )
                        self.matcher(
                            path1,
                            path2,
                            mkpt_storage,
                            cropper=cropper,
                            image_reader=scene.get_image,
                        )

                keypoint_storage = InMemoryKeypointStorage()
                matching_storage = InMemoryMatchingStorage()
                mkpt_storage.to_keypoints_and_matches(
                    keypoint_storage=keypoint_storage,
                    matching_storage=matching_storage,
                    apply_round=self.conf.round_matched_keypoints,
                )
            elif self.conf.matching_stage_mode == "hybrid_matcher_override":
                # Mode: "hybrid_matcher_override"
                # -------------------------------
                # 1. Matched keypoints from the clustering stage will not be used
                # 2. MASt3RHybridMatcher computes matchings for each pair
                # Required matchers:
                #   - MASt3RHybridMatcher
                assert self.hybrid_matcher is not None
                assert self.conf.matcher_hybrid
                _mk_storage = InMemoryMatchedKeypointStorage()  # No used
                keypoint_storage = InMemoryKeypointStorage()
                matching_storage = InMemoryMatchingStorage()
                run_point_tracking_matching(
                    self.hybrid_matcher,
                    pairs,
                    clustered_scene,
                    keypoint_storage,
                    matching_storage,
                    _mk_storage,
                    impl_version=self.conf.matcher_hybrid.impl_version,
                    apply_round=self.conf.matcher_hybrid.apply_round,
                    mkpts_decoupling_method="imc2023",
                    matching_filter_conf=self.conf.matcher_hybrid.matching_filter,
                    progress_bar=iterator,
                )
            else:
                raise ValueError(self.conf.matching_stage_mode)

            # Add keypoints and matches into COLMAP DB
            database_path = str(clustered_scene.database_path)
            log(
                f"[{clustered_scene.scene}; {cluster_idx + 1}/{num_clusters}] "
                f"COLMAP database path: {database_path}"
            )
            id_mappings = import_into_colmap(
                clustered_scene,
                keypoint_storage,
                matching_storage,
                database_path=database_path,
                camera_model=self.conf.reconstruction.get_camera_model(
                    unique_resolution_num=clustered_scene.get_unique_resolution_num()
                ),
            )

            if len(keypoint_storage.keypoints) == 0:
                # Avoid COLMAP errors
                print("Outlier scene")
                cluster_results, _ = get_outlier_reconstructions(clustered_scene)
                clustered_scene_results.append(cluster_results)
                continue

            if len(matching_storage.matches) == 0:
                # Avoid COLMAP errors
                print("Outlier scene")
                cluster_results, _ = get_outlier_reconstructions(clustered_scene)
                clustered_scene_results.append(cluster_results)
                continue

            # Add two-view geometry into COLMAP DB
            g_storage = verify_matches(
                clustered_scene,
                self.conf.verification,
                keypoint_storage=keypoint_storage,
                matching_storage=matching_storage,
                id_mappings=id_mappings,
                progress_bar=iterator,
            )

            if save_snapshot:
                SceneSnapshot(
                    clustered_scene,
                    keypoint_storage,
                    matching_storage,
                    two_view_geometry_storage=g_storage,
                ).save(pipeline_id=self.pipeline_id)

            maps: dict[int, pycolmap.Reconstruction] = {}
            if self.conf.use_glomap:
                args = [
                    "glomap",
                    "mapper",
                    "--database_path",
                    database_path,
                    "--image_path",
                    str(scene.image_dir),
                    "--output_path",
                    str(scene.reconstruction_dir),
                ]
                glomap_process = subprocess.Popen(args)
                glomap_process.wait()

                if glomap_process.returncode != 0:
                    print(
                        f"Subprocess Error (Return code: {glomap_process.returncode} )"
                    )
                else:
                    maps = {
                        0: pycolmap.Reconstruction(str(scene.reconstruction_dir / "0"))
                    }

            if len(maps) == 0:
                # NOTE
                # (From https://www.kaggle.com/code/eduardtrulls/imc-2023-submission-example/notebook)
                # By default colmap does not generate a reconstruction
                # if less than 10 images are registered. Lower it to 3.
                mapper_options = pycolmap.IncrementalPipelineOptions()
                mapper_options.num_threads = 1
                mapper_options.min_model_size = (
                    self.conf.reconstruction.mapper_min_model_size or 3
                )
                if self.conf.reconstruction.mapper_max_num_models is not None:
                    mapper_options.max_num_models = (
                        self.conf.reconstruction.mapper_max_num_models
                    )
                if self.conf.reconstruction.mapper_multiple_models is not None:
                    mapper_options.multiple_models = (
                        self.conf.reconstruction.mapper_multiple_models
                    )
                if self.conf.reconstruction.mapper_min_num_matches is not None:
                    mapper_options.min_num_matches = (
                        self.conf.reconstruction.mapper_min_num_matches
                    )
                if self.conf.reconstruction.mapper_filter_max_reproj_error is not None:
                    mapper_options.mapper.filter_max_reproj_error = (
                        self.conf.reconstruction.mapper_filter_max_reproj_error
                    )
                if self.conf.reconstruction.set_scene_graph_center_node_to_init_image_id1:
                    image_id1 = get_image_id_of_scene_graph_center(
                        clustered_scene, database_path=database_path
                    )
                    if image_id1 is not None:
                        mapper_options.init_image_id1 = image_id1

                # NOTE
                # Doc: https://github.com/colmap/pycolmap/blob/master/pipeline/sfm.cc
                maps = pycolmap.incremental_mapping(
                    database_path=database_path,
                    image_path=str(clustered_scene.image_dir),
                    output_path=str(clustered_scene.reconstruction_dir),
                    options=mapper_options,
                )

            cluster_results, clustered_scene_infos = get_best_reconstruction(
                maps, clustered_scene
            )

            if SAVE_CAMERA_DEBUG_INFO:
                print(clustered_scene_infos)
                save_camera_debug_info(
                    cluster_results,
                    clustered_scene,
                    Path(f"extra/camvis/{self.pipeline_id}"),
                    prefix_dict=clustered_scene_infos["localization_by"],
                )

            clustered_scene_results.append(cluster_results)

        scene.release_all()

        outputs = {}
        image_count = 0
        for clustered_scene, cluster_results in zip(
            clustered_scenes, clustered_scene_results
        ):
            image_count += len(list(cluster_results.keys()))
            outputs[clustered_scene.scene] = copy.deepcopy(cluster_results)

        assert image_count == len(scene.image_paths)
        return outputs

    def make_pairs(
        self,
        scene: Scene,
        pairwise_scores: np.ndarray,
        basic_pair_topk: int | None = None,
        iterator: tqdm.tqdm | None = None,
    ) -> list[tuple[int, int]]:
        assert len(scene.image_paths) == len(pairwise_scores)
        assert scene.indices_in_parent_scene is not None

        basic_pairs = make_pairs_from_pairwise_scores(
            pairwise_scores, topk=basic_pair_topk
        )
        suppl_pairs = self.shortlist_generator(
            scene,
            progress_bar=iterator,
        )

        stats = {
            "basic_pair_count": len(basic_pairs),
            "suppl_pair_cand_count": len(suppl_pairs),
            "added_pair_count_from_suppl_pairs": 0,
        }

        pairs = set(basic_pairs)
        for pair in suppl_pairs:
            if pair[0] > pair[1]:
                pair = (pair[1], pair[0])

            if pair in basic_pairs:
                continue

            i, j = pair
            score = pairwise_scores[i, j]
            if score >= 0:
                # score>=0 means that (i, j) has been checked by pre-matching
                continue
            pairs.add(pair)
            stats["added_pair_count_from_suppl_pairs"] += 1

        log(f"[{scene.scene}] make_pairs: {stats}")
        return sorted(list(pairs))


def focus_on_cluster(clustered_scene: Scene, pairwise_scores: np.ndarray) -> np.ndarray:
    assert clustered_scene.indices_in_parent_scene is not None
    keeps = clustered_scene.indices_in_parent_scene
    return pairwise_scores[keeps][:, keeps].copy()


def make_pairs_from_pairwise_scores(
    pairwise_scores: np.ndarray,
    topk: int | None = None,
) -> list[tuple[int, int]]:
    pairs = set()
    for i in range(len(pairwise_scores)):
        ranks = np.argsort(-pairwise_scores[i])
        if topk is not None:
            ranks = ranks[:topk]
        ranked_scores = np.take_along_axis(pairwise_scores[i], ranks, axis=0)

        keeps = ranked_scores > 0
        ranks = ranks[keeps]
        ranked_scores = ranked_scores[keeps]
        # print(ranked_scores)

        for j in ranks:
            if i < j:
                pairs.add((i, j))
            else:
                pairs.add((j, i))
    return sorted(list(pairs))


def get_best_reconstruction(
    maps: dict[int, pycolmap.Reconstruction],
    scene: Scene,
) -> tuple[dict, dict]:
    images_registered = 0
    best_idx = None
    for idx, rec in maps.items():
        print(idx, rec.summary())
        if len(rec.images) > images_registered:
            images_registered = len(rec.images)
            best_idx = idx

    if best_idx is None:
        return get_outlier_reconstructions(scene)

    results = {}
    infos = {"localization_by": {}}
    camid_im_map = {}
    for k, im in maps[best_idx].images.items():
        key = scene.data_schema.format_output_key(
            scene.dataset,
            scene.scene,
            im.name,
        )
        metadata = scene.data_schema.get_output_metadata(
            scene.dataset,
            scene.scene,
            im.name,
        )
        results[key] = {
            "R": copy.deepcopy(im.cam_from_world.rotation.matrix()),
            "t": copy.deepcopy(np.array(im.cam_from_world.translation)),
            "metadata": metadata,
        }
        infos["localization_by"][key] = "colmap"
        camid_im_map[im.camera_id] = im.name

    try:
        for idx, rec in maps.items():
            u_cameras = []
            g_cameras = []
            if idx == best_idx:
                continue

            for k, im in rec.images.items():
                key = scene.data_schema.format_output_key(
                    scene.dataset,
                    scene.scene,
                    im.name,
                )
                if key in results:
                    g_R = copy.deepcopy(results[key]["R"])
                    g_t = copy.deepcopy(results[key]["t"])
                    g_C = -g_R.T @ g_t

                    u_R = copy.deepcopy(im.cam_from_world.rotation.matrix())
                    u_t = copy.deepcopy(np.array(im.cam_from_world.translation))
                    u_C = -u_R.T @ u_t
                    g_cameras.append(g_C.reshape(3, 1))
                    u_cameras.append(u_C.reshape(3, 1))
            if len(g_cameras) < 3:
                print(
                    f"# of cameras that are registered to both rec({idx}) and best({best_idx}): {len(g_cameras)}"
                )
                continue
            g_cameras = np.array(g_cameras).reshape(3, -1)
            u_cameras = np.array(u_cameras).reshape(3, -1)
            inl_cf = 0
            strict_cf = -1
            thresholds = np.array([0.025, 0.05, 0.1, 0.2, 0.5, 1.0])
            model = register_by_Horn(
                u_cameras, g_cameras, np.asarray(thresholds), inl_cf, strict_cf
            )
            T = np.squeeze(model["transf_matrix"][-1])
            # print(T)
            # print(T[:3].shape)
            for k, im in rec.images.items():
                key = scene.data_schema.format_output_key(
                    scene.dataset,
                    scene.scene,
                    im.name,
                )
                if key not in results:
                    Tcw2 = np.eye(4)
                    Tcw2[:3, :3] = copy.deepcopy(im.cam_from_world.rotation.matrix())
                    Tcw2[:3, 3] = copy.deepcopy(np.array(im.cam_from_world.translation))
                    Tw2c = np.linalg.inv(Tcw2)
                    Tw1c = np.matmul(T, Tw2c)
                    Tcw1 = np.linalg.inv(Tw1c)
                    results[key]["R"] = copy.deepcopy(Tcw1[:3, :3])
                    results[key]["t"] = copy.deepcopy(Tcw1[:3, 3])
                    infos["localization_by"][key] = "horn"
                    print(f"Registered {key} by alignment to the best reconstruction")
                else:
                    print(
                        f"Registered {key}, but it has already been in the best reconstruction"
                    )
    except Exception as e:
        print(f"Registration failed: {e}")

    # Failures_to_outliers:
    for path in scene.image_paths:
        key1 = scene.data_schema.format_output_key(
            scene.dataset, scene.scene, Path(path).name
        )
        if key1 not in results:
            print(
                f"Reconstruction failed: "
                f"{scene}[{key1}] -> {DEFAULT_OUTLIER_SCENE_NAME}"
            )
            metadata = scene.data_schema.get_output_metadata(
                scene.dataset,
                scene.scene,
                Path(path).name,
            )
            R = np.eye(3) * np.nan
            t = np.zeros(3) * np.nan
            results[key1] = {
                "R": R,
                "t": t,
                "cluster_name": DEFAULT_OUTLIER_SCENE_NAME,
                "metadata": metadata,
            }
            infos["localization_by"][key1] = "fill_nan"

    return results, infos
