from __future__ import annotations

import collections
import gc
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import sklearn.cluster
import torch
from vggt.models.vggt import VGGT

from clusterings.base import Clustering, ClusteringResult
from clusterings.config import MASt3RFPSClusteringConfig
from data import resolve_model_path
from global_descriptors.mast3r_retrieval_spoc import (
    MASt3RRetrievalSPoCGlobalDescriptorExtractor,
)
from matchers.mast3r import MASt3RMatcher
from matchers.vggt import load_and_preprocess_images_imc25
from shortlists.global_descriptor import (
    create_global_descriptor_extractor,
    extract_global_features,
)
from storage import InMemoryMatchedKeypointStorage


class MASt3RFPSClustering(Clustering):
    def __init__(self, conf: MASt3RFPSClusteringConfig, device: torch.device):
        self.conf = conf
        self.extractor = MASt3RRetrievalSPoCGlobalDescriptorExtractor(
            conf.mast3r_retrieval_model, device=device
        )
        self.matcher = MASt3RMatcher(self.conf.mast3r_matcher, device=device)

        self.device = device
        self.rng = np.random.default_rng(2025)

    @torch.inference_mode()
    def run(
        self,
        image_paths: list[str | Path],
        image_reader: Callable | None = None,
        pre_computed_pairs: list[tuple[int, int]] | None = None,
        neighbor_metric: Literal["dist", "pair"] = "dist",
    ) -> ClusteringResult:
        all_indices = np.arange(len(image_paths))
        N = len(all_indices)
        print("MASt3RFPSClustering starts")
        print(f"MASt3RFPSClustering | {pre_computed_pairs=}, {neighbor_metric=}")
        self._log(f"Given {N} images")

        matched_keypoint_storage = InMemoryMatchedKeypointStorage()

        _feats = extract_global_features(
            image_paths,
            self.extractor,
            batch_size=1,
        )
        _dists = torch.cdist(_feats, _feats, p=2)

        feats = _feats.cpu().numpy()
        dists = _dists.cpu().numpy()
        initial_point = self.get_initial_point(feats, dists)
        self._log(f"Initial point: {initial_point}")

        fps_indices, fps_dists = farthest_point_sampling(
            dists,
            self.rng,
            initial_point=initial_point,
            N=self.conf.fps_n,
            dist_thresh=self.conf.fps_dist_threshold,
        )
        self._log(f"FPS.indices: {fps_indices}")
        self._log(f"FPS.dists: {fps_dists}")

        # Initialize all scores with -1
        #  - score=-1 means that matching the pair has not been computed yet
        #  - score=0 means that matching the pair has been already computed, but it's not matched
        #  - score>0 means that matching the pair has been already computed, and it's matched with the score
        pairwise_scores = np.zeros((N, N), dtype=np.int64) - 1

        # Initialize all labels with 0
        cluster_labels = np.zeros((N,), dtype=np.int64)

        # Set indices that are corresponding to samples that FPS returned to 1, 2, ...
        cluster_labels[fps_indices] = np.arange(len(fps_indices), dtype=np.int64) + 1
        self._log(f"Initial cluster labels: {cluster_labels}")

        cursors = [index for index in fps_indices]
        used_queries = set(cursors)
        waitings = set(all_indices) - set(fps_indices)

        depth = 0
        finished = False
        while len(waitings) > 0 and not finished:
            depth += 1
            self._log("----------------------")
            self._log(f"Stage({depth}) starts")
            self._log(f"# of waitings: {len(waitings)}")
            self._log(f"# of cursors: {len(cursors)}")
            self._log("")
            self._log("** FPS-side query phase **")
            self._log(f"Cursors: {cursors}")

            if len(cursors) == 0:
                self._log(f"No cursors. Break at the begin of stage {depth}")
                break

            new_cursors = set()
            for i, query_id in enumerate(cursors):
                self._log(f"Stage({depth}; phase=FPS-side)::query({query_id}) starts")
                self._log(f" * query_image = {Path(image_paths[query_id]).name}")
                self._log_cluster_sizes(cluster_labels)
                self._log(cluster_labels)
                cluster_label_to_ids = self._make_cluster_label_map(cluster_labels)
                self._log(cluster_label_to_ids)

                if len(cluster_labels[cluster_labels == 0]) == 0:
                    finished = True
                    self._log("All samples have been assigned")
                    break

                if depth <= self.conf.limit_depth_for_exhaustive_matching:
                    topk = None
                else:
                    topk = self.conf.topk_for_partial_matching

                accepts, rejects = self.search_step(
                    image_paths,
                    query_id,
                    dists,
                    matched_keypoint_storage,
                    pairwise_scores,
                    topk=topk,
                    pairs=pre_computed_pairs,
                    neighbor_metric=neighbor_metric,
                )
                self._log(f" * accepted: {accepts}")
                self._log(
                    f" * accepted images = {[Path(image_paths[ti]).name for ti in accepts]}"
                )
                self._log(f" * rejected: {rejects}")
                self._log(
                    f" * rejected images = {[Path(image_paths[ti]).name for ti in rejects]}"
                )
                if len(accepts) == 0:
                    self._log(f"Cursor[{i}](q={query_id}): Reject all neighbors")
                    new_label = cluster_labels.max() + 1
                    cluster_labels[query_id] = new_label
                    continue

                self._log(
                    f"Cursor[{i}](q={query_id}): Accepts {len(accepts)} neighbors"
                )

                query_label = cluster_labels[query_id]
                matched_target_labels = cluster_labels[accepts]
                self._log(
                    f"Existing labels of accepted targets: {matched_target_labels}"
                )
                label_for_merging = np.unique(
                    [query_label] + matched_target_labels.tolist()
                ).max()

                if query_label != label_for_merging:
                    new_label = label_for_merging
                    self._log(
                        f"Label conflict: "
                        f"query_label({query_label}) != label_for_merging({label_for_merging})"
                    )
                else:
                    new_label = query_label

                # Set new label to accepted indices
                cluster_labels[accepts] = new_label

                # Set new label to the query if needed
                cluster_labels[query_id] = new_label

                # Propagate new label
                for label_to_replace in [query_label] + matched_target_labels.tolist():
                    if label_to_replace == 0 or label_to_replace == -1:
                        continue
                    indices_to_replace = cluster_label_to_ids[label_to_replace]
                    cluster_labels[indices_to_replace] = new_label

                waitings = waitings - set(accepts)
                for target_id in accepts:
                    if int(target_id) not in used_queries:
                        new_cursors.add(int(target_id))
                        used_queries.add(int(target_id))

            self._log("")
            self._log("** Target-side query phase **")
            queries = np.array([], dtype=np.int64)
            if self.conf.num_query_from_target > 0:
                cluster_label_to_ids = self._make_cluster_label_map(cluster_labels)
                if 0 in cluster_label_to_ids:
                    queries = np.array(cluster_label_to_ids[0], dtype=np.int64)
                    queries = self.rng.permutation(queries)[
                        : self.conf.num_query_from_target
                    ]

            self._log(f"Target-side queries: {queries}")
            for i, query_id in enumerate(queries):
                self._log(
                    f"Stage({depth}; phase=target-side)::query({query_id}) starts"
                )
                self._log(f" * query_image = {Path(image_paths[query_id]).name}")
                self._log_cluster_sizes(cluster_labels)
                self._log(cluster_labels)
                cluster_label_to_ids = self._make_cluster_label_map(cluster_labels)
                self._log(cluster_label_to_ids)

                if len(cluster_labels[cluster_labels == 0]) == 0:
                    finished = True
                    self._log("All samples have been assigned")
                    break

                topk = self.conf.topk_for_partial_matching
                accepts, rejects = self.search_step(
                    image_paths,
                    query_id,
                    dists,
                    matched_keypoint_storage,
                    pairwise_scores,
                    topk=topk,
                    pairs=pre_computed_pairs,
                    neighbor_metric=neighbor_metric,
                )
                self._log(f" * accepted: {accepts}")
                self._log(
                    f" * accepted images = {[Path(image_paths[ti]).name for ti in accepts]}"
                )
                self._log(f" * rejected: {rejects}")
                self._log(
                    f" * rejected images = {[Path(image_paths[ti]).name for ti in rejects]}"
                )
                if len(accepts) == 0:
                    self._log(f"Cursor[{i}](q={query_id}): Reject all neighbors")
                    new_label = cluster_labels.max() + 1
                    cluster_labels[query_id] = new_label
                    continue

                self._log(
                    f"Cursor[{i}](q={query_id}): Accepts {len(accepts)} neighbors"
                )

                query_label = cluster_labels[query_id]
                matched_target_labels = cluster_labels[accepts]
                self._log(
                    f"Existing labels of accepted targets: {matched_target_labels}"
                )
                label_for_merging = np.unique(
                    [query_label] + matched_target_labels.tolist()
                ).max()

                if query_label != label_for_merging:
                    new_label = label_for_merging
                    self._log(
                        f"Label conflict: "
                        f"query_label({query_label}) != label_for_merging({label_for_merging})"
                    )
                else:
                    new_label = query_label

                # Set new label to accepted indices
                cluster_labels[accepts] = new_label

                # Set new label to the query if needed
                cluster_labels[query_id] = new_label

                # Propagate new label
                for label_to_replace in [query_label] + matched_target_labels.tolist():
                    if label_to_replace == 0 or label_to_replace == -1:
                        continue
                    indices_to_replace = cluster_label_to_ids[label_to_replace]
                    cluster_labels[indices_to_replace] = new_label

                waitings = waitings - set(accepts)

            self._log("")
            self._log(f"stage({depth}) done")
            self._log(f"{len(new_cursors)} queries will be added to cursors")
            cursors = sorted(list(new_cursors))
            self._log(f"Used queries: {used_queries}")
            self._log(f"Next queries: {cursors}")

        self._log("Incremental clustering finished")
        self._log(f"cluster_labels: {cluster_labels}")

        if len(cluster_labels[cluster_labels == 0]) > 0:
            cluster_labels[cluster_labels == 0] = -1
            self._log("Replace 0 with -1")

        cluster_label_to_ids = self._make_cluster_label_map(cluster_labels)
        for label, _indices in cluster_label_to_ids.items():
            if label == -1:
                continue
            if len(_indices) < self.conf.min_cluster_size:
                self._log(f"Small cluster: label({label}) -> -1")
                cluster_labels[_indices] = -1

        if self.conf.use_noisy_cluster_as_one_cluster:
            new_cluster_label = cluster_labels.max() + 1
            cluster_labels[cluster_labels == -1] = new_cluster_label

        # Re-index
        cluster_label_to_ids = self._make_cluster_label_map(cluster_labels)
        final_cluster_labels = np.zeros_like(cluster_labels)
        if -1 in cluster_label_to_ids:
            noisy_indices = cluster_label_to_ids[-1]
            final_cluster_labels[noisy_indices] = -1
            cluster_label_to_ids.pop(-1)
        for new_label, current_label in enumerate(sorted(cluster_label_to_ids.keys())):
            indices = cluster_label_to_ids[current_label]
            final_cluster_labels[indices] = new_label

        print("MASt3RFPSClustering finished")
        print(f"final_cluster_labels: {final_cluster_labels}")
        return (
            ClusteringResult(image_paths, final_cluster_labels)
            .add_output("matched_keypoint_storage", matched_keypoint_storage)
            .add_output("global_feature", feats.copy())
            .add_output("pairwise_distance", dists.copy())
            .add_output("pairwise_score", pairwise_scores.copy())
        )

    def search_step(
        self,
        image_paths: list[str | Path],
        query_id: int,
        dists: np.ndarray,
        storage: InMemoryMatchedKeypointStorage,  # For storing internal results
        pairwise_score_table: np.ndarray,  # For storing and refering pairwise scores
        topk: int | None = None,
        pairs: list[tuple[int, int]] | None = None,
        neighbor_metric: Literal["dist", "pair"] = "dist",
    ) -> tuple[np.ndarray, np.ndarray]:
        if neighbor_metric == "dist":
            dists_q_vs_all = dists[query_id]
            ranks = dists_q_vs_all.argsort()
            if topk is None:
                self._log(f"search_step(q={query_id}): {ranks}")
                target_indices = ranks[1:]
            else:
                self._log(f"search_step(q={query_id}): {ranks[: topk + 1]}")
                target_indices = ranks[1 : topk + 1]
        elif neighbor_metric == "pair":
            self._log("neighbor_metric=pair ignores topk")
            assert pairs is not None
            _target_indices = []
            for i, j in pairs:
                if i == query_id:
                    _target_indices.append(j)
                    continue
                if j == query_id:
                    _target_indices.append(i)
                    continue
            target_indices = np.array(sorted(_target_indices))
            self._log(f"search_step(q={query_id}): {target_indices}")
        else:
            raise ValueError(neighbor_metric)

        self._log(f"len={len(target_indices)}")

        matching, scores = self.matching_step(
            image_paths,
            query_id,
            target_indices,
            storage,
            pairwise_score_table,
        )
        accepts = target_indices[matching]
        rejects = target_indices[~matching]

        # Update the pairwise score table
        assert len(scores) == len(target_indices)
        for target_id, score in zip(target_indices, scores):
            pairwise_score_table[query_id, target_id] = score
            pairwise_score_table[target_id, query_id] = score

        return accepts, rejects

    def get_initial_point(self, feats: np.ndarray, dists: np.ndarray) -> int | None:
        if self.conf.initial_point_type == "simsum_max":
            sims = -dists
            sim_sums = sims.sum(axis=1)
            most_common_id = np.argmax(sim_sums)
            return int(most_common_id)
        elif self.conf.initial_point_type in ("dbscan", "dbscan-bug"):
            if self.conf.initial_point_type == "dbscan-bug":
                print(
                    "[BUG] initial_point_type='dbscan-bug' for backward compatibility"
                )
                feats = dists

            try:
                dbscan = sklearn.cluster.DBSCAN(
                    eps=0.3,
                    min_samples=5,
                ).fit(feats)
                labels = dbscan.labels_.copy()
            except Exception as e:
                self._log("DBSCAN failed: initial point -> None")
                return None

            num_clusters = len(np.unique(labels))
            if num_clusters < 2:
                self._log(
                    f"DBSCAN returns {num_clusters} clusters: initial point -> None"
                )
                return None

            noisy, *_ = np.where(labels == -1)
            cands, *_ = np.where(labels != -1)
            if len(noisy) > len(cands):
                self._log("DBSCAN returns noisy clusters: initial point -> None")
                return None

            counts = np.zeros((num_clusters,), dtype=np.int32)
            cand_dict: dict[int, list[int]] = collections.defaultdict(list)
            for i in cands:
                label = int(labels[i])
                assert label != -1
                counts[label] += 1
                cand_dict[label].append(i)
            self._log(f"Pre-clustering by DBSCAN: {cand_dict}")
            largest_id = int(counts.argmax())
            self._log(f"Largest cluster id: {largest_id}")
            return int(self.rng.choice(cand_dict[largest_id], 1))
        else:
            raise ValueError(self.conf.initial_point_type)

    def matching_step(
        self,
        image_paths: list[str | Path],
        query_id: int,
        target_indices: np.ndarray,
        storage: InMemoryMatchedKeypointStorage,
        pairwise_score_table: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mast3r_matches = []
        mast3r_scores = []
        self._log(
            f"MASt3R try match {query_id} vs {target_indices} ({len(target_indices)})"
        )
        for target_id in target_indices:
            if pairwise_score_table[query_id, target_id] >= 0:
                score = pairwise_score_table[query_id, target_id]
                if score > 0:
                    mast3r_matches.append(True)
                else:
                    mast3r_matches.append(False)
                mast3r_scores.append(score)
                # (q, t) has a score, so skip matching
                continue

            if pairwise_score_table[target_id, query_id] >= 0:
                score = pairwise_score_table[target_id, query_id]
                if score > 0:
                    mast3r_matches.append(True)
                else:
                    mast3r_matches.append(False)
                mast3r_scores.append(score)
                # (t, q) has a score, so skip matching
                continue

            path1 = str(image_paths[query_id])
            path2 = str(image_paths[target_id])
            self.matcher(path1, path2, storage)
            if storage.has(path1, path2):
                mkpts1, _ = storage.get(path1, path2)
                # TODO
                min_matches = self.conf.mast3r_matcher.min_matches
                if min_matches is None:
                    min_matches = 15

                score = len(mkpts1)
                if score >= min_matches:
                    mast3r_matches.append(True)
                else:
                    score = 0
                    mast3r_matches.append(False)
            else:
                score = 0
                mast3r_matches.append(False)
            mast3r_scores.append(score)

        self._log(mast3r_matches)
        return (
            np.array(mast3r_matches, dtype=np.bool_),
            np.array(mast3r_scores, dtype=np.int64),
        )

    def _make_cluster_label_map(
        self, cluster_labels: np.ndarray
    ) -> dict[int, list[int]]:
        cluster_label_to_ids = collections.defaultdict(list)
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


def farthest_point_sampling(
    dist: np.ndarray,
    rng: np.random.Generator,
    initial_point: int | None,
    N: int | None = None,
    dist_thresh: float | None = None,
):
    """Farthest point sampling.
    from mast3r.retrieval.graph import farthest_point_sampling
    """

    assert N is not None or dist_thresh is not None, (
        "Either N or min_dist must be provided."
    )

    if N is None:
        N = dist.shape[0]

    indices = []
    distances = [0]
    if initial_point is None:
        indices.append(rng.choice(dist.shape[0]))
    else:
        indices.append(initial_point)

    for i in range(1, N):
        d = dist[indices].min(axis=0)
        bst = d.argmax()
        bst_dist = d[bst]
        if dist_thresh is not None and bst_dist < dist_thresh:
            break
        indices.append(bst)
        distances.append(bst_dist)
    return np.array(indices), np.array(distances)
