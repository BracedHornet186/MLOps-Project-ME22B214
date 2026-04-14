from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from clusterings.base import Clustering, ClusteringResult


class DebugArraySplitClustering(Clustering):
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters

    def run(
        self,
        image_paths: list[str | Path],
        image_reader: Callable | None = None,
    ) -> ClusteringResult:
        labels = np.zeros(len(image_paths), dtype=np.int64)
        for a, k in zip(
            np.array_split(labels, self.n_clusters), range(self.n_clusters)
        ):
            a += k
        return ClusteringResult(
            image_paths=image_paths,
            cluster_labels=labels,
        )
