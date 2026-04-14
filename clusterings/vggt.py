from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from vggt.models.vggt import VGGT

from clusterings.base import Clustering, ClusteringResult
from clusterings.config import VGGTClusteringConfig
from data import resolve_model_path
from matchers.vggt import load_and_preprocess_images_imc25


class VGGTClustering(Clustering):
    def __init__(self, conf: VGGTClusteringConfig, device: torch.device):
        self.conf = conf
        self.model = (
            VGGT.from_pretrained(resolve_model_path(conf.model.pretrained_model))
            .eval()
            .to(device)
        )
        self.device = device
        self.rng = np.random.default_rng(self.conf.seed)
        assert (
            self.conf.depth_score_threshold is not None
            or self.conf.world_points_score_threshold is not None
        )
        self.seen_count_table = np.empty((0, 0), dtype=np.int32)

    def init_state(self, image_paths: list[str | Path]):
        N = len(image_paths)
        self.seen_count_table = np.zeros((N, N), dtype=np.int32)
        self._log(f"Seen count table is initialized with ({N}, {N})")

    @torch.inference_mode()
    def run(
        self,
        image_paths: list[str | Path],
        image_reader: Callable | None = None,
    ) -> ClusteringResult:
        self.init_state(image_paths)

        result = None
        for stage_result in self.run_stage(image_paths):
            result = stage_result
        assert result is not None
        return result

    def run_stage(self, image_paths: list[str | Path]) -> Generator[ClusteringResult]:
        # Initialize all labels with -1
        cluster_labels = np.zeros((len(image_paths),), dtype=np.int64) - 1

        for i in range(self.conf.num_cycles):
            self._log("==============")
            self._log(f"Stage {i + 1}")
            self._log("==============")
            self._log(f"Scan matching stage ({i + 1}/{self.conf.num_cycles}) starts")
            result = self.scan_matching_stage(image_paths, cluster_labels)

            self._log("------")
            self._log(f"Cluster matching stage ({i + 1}/{self.conf.num_cycles})")
            cluster_labels = result.cluster_labels.copy()
            result = self.cluster_matching_stage(image_paths, cluster_labels)

            yield result

            cluster_labels = result.cluster_labels.copy()

    def _record_seen_pair(
        self, query_id: int, target_indices: np.ndarray, count_inv_pair: bool = False
    ):
        for i, j in zip([query_id] * len(target_indices), target_indices):
            self.seen_count_table[i, j] += 1
            if count_inv_pair:
                # NOTE: count_inv_pair=False makes results slightly better
                self.seen_count_table[j, i] += 1

    def _get_scan_matching_candidates(
        self,
        image_paths: list[str | Path],
        query_id: int,
        query_cluster_chained_indices: np.ndarray,
    ):
        N = len(image_paths)
        all_indices = np.arange(N)
        cand_indices = all_indices[all_indices != query_id]  # Filter out query id
        cand_indices = cand_indices[
            ~np.isin(cand_indices, query_cluster_chained_indices)
        ]

        seen_counts_q_vs_all = self.seen_count_table[query_id]  # Shape(N,)
        seen_counts_q_vs_cands = seen_counts_q_vs_all[cand_indices]

        print(seen_counts_q_vs_all)
        print(seen_counts_q_vs_cands)
        cand_indices = cand_indices[seen_counts_q_vs_cands == 0]
        print(f"#cands = {len(cand_indices)}")

        return cand_indices

    def scan_matching_stage(
        self,
        image_paths: list[str | Path],
        cluster_labels: np.ndarray,
    ) -> ClusteringResult:
        N = len(image_paths)
        new_cluster_labels = cluster_labels.copy()
        for query_id in range(len(image_paths)):
            cluster_label_to_ids = self._make_cluster_label_map(new_cluster_labels)
            self._log_cluster_sizes(new_cluster_labels)
            self._log(cluster_label_to_ids)

            query_label = new_cluster_labels[query_id]
            self._log(f" * query_id = {query_id}")
            self._log(f" * query_label = {query_label}")
            self._log(f" * query_image = {Path(image_paths[query_id]).name}")

            if query_label == -1:
                query_cluster_chained_indices = np.array([], dtype=np.int64)
            else:
                query_cluster_chained_indices = np.array(
                    cluster_label_to_ids[query_label], dtype=np.int64
                )
            cand_indices = self._get_scan_matching_candidates(
                image_paths,
                query_id,
                query_cluster_chained_indices,
            )

            if len(cand_indices) == 0:
                break

            # Sample randomly
            target_indices = self.rng.choice(
                cand_indices,
                min(self.conf.window_size - 1, len(cand_indices)),
                replace=False,
            )
            self._record_seen_pair(query_id, target_indices)

            target_evaluations = self.matching_step(
                image_paths, query_id=query_id, target_indices=target_indices
            )
            self._log(f" * evaluations: {target_evaluations}")
            self._log(f" * target_indices = {target_indices}")
            self._log(
                f" * target_images = {[Path(image_paths[ti]).name for ti in target_indices]}"
            )
            if target_evaluations.sum() == 0:
                continue

            hit_target_indices = target_indices[target_evaluations]
            hit_target_labels = new_cluster_labels[hit_target_indices]

            self._log(f" * hit_target_indices = {hit_target_indices}")
            self._log(f" * hit_target_labels = {hit_target_labels}")
            label_for_merging = np.unique(
                [query_label] + hit_target_labels.tolist()
            ).max()
            self._log(f" * label_for_merging = {label_for_merging}")
            if label_for_merging == -1:
                # Case: All matched targets are assinged to -1
                # --------------------------------------------

                # Assign the new label to the query and matched targets
                new_cluster_label = int(new_cluster_labels.max()) + 1
                new_cluster_labels[query_id] = new_cluster_label
                new_cluster_labels[hit_target_indices] = new_cluster_label

                if len(query_cluster_chained_indices) > 0:
                    # If the query has been in a cluster, propagate the new label to them
                    new_cluster_labels[query_cluster_chained_indices] = (
                        new_cluster_label
                    )
            else:
                # Case: Some label is already assigned to one of the query or targets
                # -------------------------------------------------------------------
                for label_to_replace in np.unique(hit_target_labels):
                    if label_to_replace == -1:
                        pass
                    elif label_to_replace == label_for_merging:
                        pass
                    else:
                        indices_to_replace = cluster_label_to_ids[label_to_replace]
                        new_cluster_labels[indices_to_replace] = label_for_merging
                new_cluster_labels[query_id] = label_for_merging
                new_cluster_labels[hit_target_indices] = label_for_merging

                if len(query_cluster_chained_indices) > 0:
                    # If the query has been in a cluster, propagate the new label to them
                    new_cluster_labels[query_cluster_chained_indices] = (
                        label_for_merging
                    )

        self._log_cluster_sizes(new_cluster_labels)
        return ClusteringResult(image_paths, new_cluster_labels)

    def cluster_matching_stage(
        self,
        image_paths: list[str | Path],
        cluster_labels: np.ndarray,
    ) -> ClusteringResult:
        N = len(image_paths)
        new_cluster_labels = cluster_labels.copy()
        initial_unique_cluster_labels = np.unique(new_cluster_labels)
        for query_label in initial_unique_cluster_labels:
            if query_label == -1:
                continue

            cluster_label_to_ids = self._make_cluster_label_map(new_cluster_labels)
            if query_label not in cluster_label_to_ids:
                continue

            self._log_cluster_sizes(new_cluster_labels)
            self._log(cluster_label_to_ids)
            self._log(f" * query_label = {query_label}")

            unique_cluster_labels = np.array(
                sorted(cluster_label_to_ids.keys()), dtype=np.int64
            )
            unique_cluster_labels = unique_cluster_labels[
                unique_cluster_labels != query_label
            ]
            unique_cluster_labels = unique_cluster_labels[unique_cluster_labels != -1]
            self._log(f" * unique_cluster_labels = {unique_cluster_labels}")
            if len(unique_cluster_labels) == 0:
                continue

            target_cluster_labels = unique_cluster_labels[
                self.rng.permutation(len(unique_cluster_labels))[
                    : self.conf.window_size - 1
                ]
            ]
            self._log(f" * target_cluster_labels = {target_cluster_labels}")
            target_indices = []
            for target_cluster_label in target_cluster_labels:
                target_indices.append(
                    int(self.rng.choice(cluster_label_to_ids[target_cluster_label], 1))
                )
            target_indices = np.array(target_indices, dtype=np.int64)

            query_id = int(self.rng.choice(cluster_label_to_ids[query_label], 1))

            self._log(f" * query_id = {query_id}")
            self._log(f" * target_indices = {target_indices}")

            target_evaluations = self.matching_step(
                image_paths, query_id=query_id, target_indices=target_indices
            )
            self._log(f" * evaluations: {target_evaluations}")
            if target_evaluations.sum() == 0:
                continue

            hit_target_labels = target_cluster_labels[target_evaluations]
            labels_to_merge = np.array(
                [query_label] + hit_target_labels.tolist(), dtype=np.int64
            )
            label_for_merging = labels_to_merge.max()
            self._log(f" * labels_to_merge: {labels_to_merge}")
            self._log(f" * label_for_merging: {label_for_merging}")
            for label_to_merge in labels_to_merge:
                new_cluster_labels[
                    np.array(cluster_label_to_ids[label_to_merge], dtype=np.int64)
                ] = label_for_merging

            self._log_cluster_sizes(new_cluster_labels)
            self._log(cluster_label_to_ids)

        self._log_cluster_sizes(new_cluster_labels)
        return ClusteringResult(image_paths, new_cluster_labels)

    def matching_step(
        self,
        image_paths: list[str | Path],
        query_id: int,
        target_indices: np.ndarray,
    ) -> np.ndarray:
        indices = np.array(
            [query_id] + target_indices.tolist(), dtype=target_indices.dtype
        )
        batch_image_paths = [image_paths[int(i)] for i in indices]
        predictions = self._predict(batch_image_paths)
        evaluations = self._evaluate(predictions, indices)
        assert len(evaluations) == len(indices)
        assert bool(evaluations[0])
        target_evaluations = evaluations[1:]
        assert len(target_evaluations) == len(target_indices)
        return target_evaluations

    def _evaluate(
        self, predictions: dict[str, np.ndarray], indices: np.ndarray
    ) -> np.ndarray:
        assert (
            len(indices)
            == len(predictions["depth_conf"])
            == len(predictions["world_points_conf"])
        )
        same_flags = []
        scores = []
        for i, index, depth_scores, world_points_scores in zip(
            range(len(indices)),
            indices,
            predictions["depth_conf"],
            predictions["world_points_conf"],
        ):
            if i == 0:
                # Assume that the first element is "query"
                same_flags.append(True)
                continue
            same_flag = False
            if self.conf.depth_score_threshold is not None:
                score = depth_scores.max()
                if score >= self.conf.depth_score_threshold:
                    same_flag = True
            elif self.conf.world_points_score_threshold is not None:
                score = world_points_scores.max()
                if score >= self.conf.world_points_score_threshold:
                    same_flag = True
            else:
                raise ValueError
            scores.append(score)
            same_flags.append(same_flag)
        self._log(f" * scores: {scores}")
        return np.array(same_flags, dtype=bool)

    def _predict(self, image_paths: list[str | Path]) -> dict[str, np.ndarray]:
        images, _, _, _ = load_and_preprocess_images_imc25(
            image_paths, mode="pad", target_size=self.conf.image_size
        )
        images = images.to(self.device)
        with torch.no_grad():
            with torch.autocast(self.device.type):
                _predictions = self.model(images)

        predictions = {}
        for key in _predictions.keys():
            if isinstance(_predictions[key], torch.Tensor):
                predictions[key] = (
                    _predictions[key].cpu().numpy().squeeze(0)
                )  # Remove batch dimension

        return predictions

    def _make_cluster_label_map(
        self, cluster_labels: np.ndarray
    ) -> dict[int, list[int]]:
        cluster_label_to_ids = defaultdict(list)
        for i in range(len(cluster_labels)):
            label = cluster_labels[i]
            cluster_label_to_ids[label].append(i)
        return cluster_label_to_ids

    def _log(self, message: Any, show: bool = True):
        if not show:
            return
        if not self.conf.verbose:
            return
        print(f"[{self.__class__.__name__}] {message}")

    def _log_cluster_sizes(self, cluster_labels: np.ndarray):
        if not self.conf.verbose:
            return
        unique_cluster_labels = np.unique(cluster_labels)
        sizes = {
            label: len(cluster_labels[cluster_labels == label])
            for label in unique_cluster_labels
        }
        self._log(f" * {len(unique_cluster_labels)} clusters: {sizes}")
