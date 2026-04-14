from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import sklearn.cluster

from clusterings.base import Clustering, ClusteringResult
from shortlists.global_descriptor import (
    ConcatDescriptorExtractor,
    DescriptorExtractor,
    extract_global_features,
)


class DBSCANClustering(Clustering):
    def __init__(
        self,
        extractor: DescriptorExtractor | ConcatDescriptorExtractor,
        batch_size: int = 16,
        eps: float = 0.5,
        min_samples: int = 5,
    ):
        self.extractor = extractor
        self.batch_size = batch_size
        self.eps = eps
        self.min_samples = min_samples

    def run(
        self,
        image_paths: list[str | Path],
        image_reader: Callable | None = None,
    ) -> ClusteringResult:
        feats = extract_global_features(
            image_paths, self.extractor, batch_size=self.batch_size
        )
        x = feats.detach().cpu().numpy()

        try:
            dbscan = sklearn.cluster.DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
            ).fit(x)
            labels = dbscan.labels_.copy()
        except Exception as e:
            labels = np.zeros((len(image_paths),), dtype=np.int64)

        return ClusteringResult(
            image_paths=image_paths,
            cluster_labels=labels,
        )
