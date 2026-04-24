from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
import tqdm

from pipelines.scene import Scene
from shortlists.base import ShortlistUpdater
from shortlists.config import NoShortlistUpdaterConfig
from shortlists.imc2024 import (
    create_global_descriptor_extractors,
    extract_global_features,
)
from scripts.storage import (
    InMemoryMatchedKeypointStorage,
    InMemoryMatchingStorage,
    Line2DSegmentStorage,
)


class NoShortlistUpdater(ShortlistUpdater):
    """Do nothing, but update topk ranks and dists
    """
    def __init__(
        self,
        conf: NoShortlistUpdaterConfig,
        device: Optional[torch.device] = None,
    ):
        self.conf = conf
        self.device = device
        self.extractors = create_global_descriptor_extractors(
            conf.global_descriptors, device=device
        )

    def to_topk_table(self, feats: torch.Tensor, topk: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        dists = torch.cdist(feats, feats, p=2).detach().cpu().float().numpy()

        ranks = np.argsort(dists)
        topk_ranks = ranks[:, : topk]
        topk_dists = np.take_along_axis(dists, topk_ranks, axis=1)

        return topk_ranks, topk_dists

    @torch.inference_mode()
    def __call__(
        self,
        scene: Scene,
        progress_bar: Optional[tqdm.tqdm] = None,
        mk_storage: Optional[InMemoryMatchedKeypointStorage] = None,
        mk_storage_list: Optional[list[InMemoryMatchedKeypointStorage]] = None,
        line2d_seg_storage: Optional[Line2DSegmentStorage] = None,
        line2d_matching_storage: Optional[InMemoryMatchingStorage] = None,
        **kwargs,
    ) -> List[Tuple[int, int]]:
        assert scene.shortlist
        print("[NoShortlistUpdater] Starts")

        if self.extractors:
            assert self.conf.global_descriptors
            global_features_list = extract_global_features(
                self.extractors,
                self.conf.global_descriptors,
                scene,
                progress_bar=progress_bar,
            )
            for feats, c in zip(global_features_list, self.conf.global_descriptors):
                topk_ranks, topk_dists = self.to_topk_table(feats)
                # NOTE:
                # When using multiple extractors,
                # ranks and dists that the last one extracts are used
                scene.update_topk_table(topk_ranks=topk_ranks, topk_dists=topk_dists)

        return scene.shortlist
