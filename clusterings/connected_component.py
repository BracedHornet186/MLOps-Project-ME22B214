from __future__ import annotations

import collections
from collections.abc import Callable
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from clusterings.base import Clustering, ClusteringResult
from retrievers.base import Retriever
from shortlists.global_descriptor import (
    ConcatDescriptorExtractor,
    DescriptorExtractor,
    extract_global_features,
)
from scripts.workspace import log


class ConnectedComponentClustering(Clustering):
    def __init__(
        self,
        extractor: DescriptorExtractor | ConcatDescriptorExtractor | Retriever,
        topk: int,
        dist_threshold: float,
        min_cluster_size: int,
        use_noisy_cluster_as_one_cluster: bool,
        degree_threshold: int | None = None,
        batch_size: int = 16,
    ):
        self.extractor = extractor
        self.topk = topk
        self.dist_threshold = dist_threshold
        self.degree_threshold = degree_threshold
        self.min_cluster_size = min_cluster_size
        self.use_noisy_cluster_as_one_cluster = use_noisy_cluster_as_one_cluster
        self.batch_size = batch_size

    @torch.inference_mode()
    def run(
        self,
        image_paths: list[str | Path],
        image_reader: Callable | None = None,
    ) -> ClusteringResult:
        if isinstance(self.extractor, DescriptorExtractor):
            feats = extract_global_features(
                image_paths, self.extractor, batch_size=self.batch_size
            )
            dists = torch.cdist(feats, feats, p=2).detach().cpu().numpy()
            ranks = np.argsort(dists)
            topk_ranks = ranks[:, : self.topk]
            topk_dists = np.take_along_axis(dists, topk_ranks, axis=1)
        elif isinstance(self.extractor, Retriever):
            topk_ranks, topk_dists = self.extractor.build(image_paths).search_nn(
                image_paths, k=self.topk, return_metric="dist"
            )
        else:
            raise TypeError

        print(f"nndist.mean: {np.mean(topk_dists[:, 1])}")
        print(f"nndist.median: {np.median(topk_dists[:, 1])}")
        print(f"nndist.5%tile: {np.percentile(topk_dists[:, 1], 5)}")
        print(f"nndist.95%tile: {np.percentile(topk_dists[:, 1], 95)}")

        xs = np.arange(len(topk_ranks))

        G = nx.Graph()
        G.add_nodes_from(xs)
        for x, _ranks, _dists in zip(xs, topk_ranks, topk_dists):
            for r, d in zip(_ranks, _dists):
                if x == r:
                    continue
                if d <= self.dist_threshold:
                    G.add_edge(int(x), int(r))

        if self.degree_threshold is not None:
            n = len(G.nodes())
            if n > self.degree_threshold + 1:
                removes = [x for x in G.nodes() if G.degree(x) <= self.degree_threshold]  # type: ignore
                G.remove_nodes_from(removes)

        labels = np.ones((len(image_paths),), dtype=np.int64) * (-1)  # outliers
        new_label = 0
        for ci, cc in enumerate(nx.connected_components(G)):
            log(f"[ConnectedComponentClustering] ci({ci}): size={len(cc)}")
            if len(cc) >= 2:
                for i in cc:
                    labels[i] = new_label
                new_label += 1
        print(labels)

        cluster_label_to_ids = self._make_cluster_label_map(labels)
        for label, _indices in cluster_label_to_ids.items():
            if label == -1:
                continue
            if len(_indices) < self.min_cluster_size:
                labels[_indices] = -1

        if self.use_noisy_cluster_as_one_cluster:
            new_cluster_label = labels.max() + 1
            labels[labels == -1] = new_cluster_label

        # Re-index
        cluster_label_to_ids = self._make_cluster_label_map(labels)
        final_cluster_labels = np.zeros_like(labels)
        for new_label, current_label in enumerate(sorted(cluster_label_to_ids.keys())):
            indices = cluster_label_to_ids[current_label]
            final_cluster_labels[indices] = new_label

        return ClusteringResult(
            image_paths=image_paths,
            cluster_labels=final_cluster_labels,
        )

    def _make_cluster_label_map(
        self, cluster_labels: np.ndarray
    ) -> dict[int, list[int]]:
        cluster_label_to_ids = collections.defaultdict(list)
        for i in range(len(cluster_labels)):
            label = cluster_labels[i]
            cluster_label_to_ids[label].append(i)
        return cluster_label_to_ids
