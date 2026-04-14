from __future__ import annotations

import collections
import gc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import sklearn.cluster
import torch
from vggt.models.vggt import VGGT

from clusterings.base import Clustering, ClusteringResult
from clusterings.config import VGGTFPSClusteringConfig
from data import resolve_model_path
from matchers.mast3r import MASt3RMatcher
from matchers.vggt import load_and_preprocess_images_imc25
from shortlists.global_descriptor import (
    create_global_descriptor_extractor,
    extract_global_features,
)
from storage import InMemoryMatchedKeypointStorage


class VGGTFPSClustering(Clustering):
    def __init__(self, conf: VGGTFPSClusteringConfig, device: torch.device):
        self.conf = conf
        self.extractor = create_global_descriptor_extractor(
            conf.global_desc,
            device=device,
        )
        self.model = (
            VGGT.from_pretrained(resolve_model_path(conf.vggt.pretrained_model))
            .eval()
            .to(device)
        )
        torch.cuda.synchronize()
        del self.model.camera_head
        del self.model.depth_head
        self.model.camera_head = None
        self.model.depth_head = None
        gc.collect()
        torch.cuda.empty_cache()

        if self.conf.mast3r_matcher:
            self.mast3r_matcher = MASt3RMatcher(self.conf.mast3r_matcher, device=device)
        else:
            self.mast3r_matcher = None

        self.device = device
        self.device = device
        self.rng = np.random.default_rng(2025)

    @torch.inference_mode()
    def run(
        self,
        image_paths: list[str | Path],
        image_reader: Callable | None = None,
    ) -> ClusteringResult:
        all_indices = np.arange(len(image_paths))
        N = len(all_indices)
        print("VGGTFPSClustering starts")
        self._log(f"Given {N} images")

        _feats = extract_global_features(
            image_paths,
            self.extractor,
            batch_size=1,
        )
        _dists = torch.cdist(_feats, _feats, p=2)

        feats = _feats.cpu().numpy()
        dists = _dists.cpu().numpy()
        initial_point = self.get_initial_point(dists)
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

        cluster_labels = np.zeros((N,), dtype=np.int64) - 1
        cluster_labels[fps_indices] = np.arange(len(fps_indices), dtype=np.int64)
        self._log(f"Initial cluster labels: {cluster_labels}")

        cursors = [index for index in fps_indices]
        used_queries = set(cursors)
        waitings = set(all_indices) - set(fps_indices)

        search_stage_count = 0
        while len(waitings) > 0:
            search_stage_count += 1
            self._log("---")
            self._log(f"Stage({search_stage_count}) starts")
            self._log(f"# of waitings: {len(waitings)}")
            self._log(f"# of cursors: {len(cursors)}")
            self._log(f"Cursors: {cursors}")

            if len(cursors) == 0:
                self._log(
                    f"No cursors. Break at the begin of stage {search_stage_count}"
                )
                break

            new_cursors = set()
            for i, query_id in enumerate(cursors):
                self._log(f"Stage({search_stage_count})::query({query_id}) starts")
                self._log(f" * query_image = {Path(image_paths[query_id]).name}")
                self._log_cluster_sizes(cluster_labels)
                self._log(cluster_labels)
                cluster_label_to_ids = self._make_cluster_label_map(cluster_labels)
                self._log(cluster_label_to_ids)

                accepts, rejects = self.search_step(image_paths, query_id, dists)
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
                    if label_to_replace == -1:
                        continue
                    indices_to_replace = cluster_label_to_ids[label_to_replace]
                    cluster_labels[indices_to_replace] = new_label

                waitings = waitings - set(accepts)
                for target_id in accepts:
                    if int(target_id) not in used_queries:
                        new_cursors.add(int(target_id))

            self._log(f"stage({search_stage_count}) done")
            self._log(f"{len(new_cursors)} queries will be added to cursors")
            for query_id in cursors:
                used_queries.add(query_id)
            cursors = sorted(list(new_cursors))
            self._log(f"Used queries: {used_queries}")
            self._log(f"Next queries: {cursors}")

        self._log("Incremental clustering finished")
        self._log(f"cluster_labels: {cluster_labels}")

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
        for new_label, current_label in enumerate(sorted(cluster_label_to_ids.keys())):
            indices = cluster_label_to_ids[current_label]
            final_cluster_labels[indices] = new_label

        print("VGGTFPSClustering finished")
        print(f"final_cluster_labels: {final_cluster_labels}")
        return ClusteringResult(image_paths, final_cluster_labels)

    def search_step(
        self,
        image_paths: list[str | Path],
        query_id: int,
        dists: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        dists_q_vs_all = dists[query_id]
        ranks = dists_q_vs_all.argsort()
        self._log(f"search_step(q={query_id}): {ranks[: self.conf.window_size]}")
        target_indices = ranks[1 : self.conf.window_size]
        matching = self.matching_step(image_paths, query_id, target_indices)
        accepts = target_indices[matching]
        rejects = target_indices[~matching]
        return accepts, rejects

    def get_initial_point(self, feats: np.ndarray) -> int | None:
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
            self._log(f"DBSCAN returns {num_clusters} clusters: initial point -> None")
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

        if self.mast3r_matcher is not None:
            non_matched_target_image_paths = [
                image_paths[int(ti)]
                for ti, is_matched in zip(target_indices, target_evaluations)
                if not is_matched
            ]
            non_matched_target_positions = [
                i for i, is_matched in enumerate(target_evaluations) if not is_matched
            ]
            mast3r_matches = []
            self._log(f"MASt3R will try re-match {non_matched_target_image_paths}")
            for path2 in non_matched_target_image_paths:
                _storage = InMemoryMatchedKeypointStorage()
                path1 = str(image_paths[query_id])
                path2 = str(path2)
                self.mast3r_matcher(path1, path2, _storage)
                if _storage.has(path1, path2):
                    mkpts1, _ = _storage.get(path1, path2)
                    # TODO
                    if len(mkpts1) >= 15:
                        mast3r_matches.append(True)
                    else:
                        mast3r_matches.append(False)
                else:
                    mast3r_matches.append(False)

            self._log(non_matched_target_positions)
            self._log(mast3r_matches)
            for i, mast3r_match in zip(non_matched_target_positions, mast3r_matches):
                self._log(
                    f"MASt3R match: {i}: {target_evaluations[i]} -> {mast3r_match} "
                )
                if mast3r_match:
                    target_evaluations[i] = mast3r_match

        return target_evaluations

    def _evaluate(
        self, predictions: dict[str, np.ndarray], indices: np.ndarray
    ) -> np.ndarray:
        assert (
            len(indices)
            # == len(predictions["depth_conf"])
            == len(predictions["world_points_conf"])
        )
        same_flags = []
        scores = []
        for i, index, world_points_scores in zip(
            range(len(indices)),
            indices,
            # predictions["depth_conf"],
            predictions["world_points_conf"],
        ):
            if i == 0:
                # Assume that the first element is "query"
                same_flags.append(True)
                continue
            same_flag = False
            # if self.conf.depth_score_threshold is not None:
            #     score = depth_scores.max()
            #     if score >= self.conf.depth_score_threshold:
            #         same_flag = True
            if self.conf.world_points_score_threshold is not None:
                score = world_points_scores.max()
                if score >= self.conf.world_points_score_threshold:
                    same_flag = True
            else:
                raise ValueError
            scores.append(score)
            same_flags.append(same_flag)
        self._log(f" * scores: {scores}")
        return np.array(same_flags, dtype=np.bool_)

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
