from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any, List, Optional, Tuple

import cv2
import networkx as nx
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as T
import tqdm

from data import FilePath
from pipelines.scene import Scene
from shortlists.base import ShortlistGenerator, get_all_pairs
from shortlists.config import ShortlistGeneratorConfig
from shortlists.global_descriptor import (
    create_global_descriptor_extractor,
    extract_global_features,
)
from workspace import log


class GlobalDescriptorConnectedComponentShortlistGenerator(ShortlistGenerator):
    def __init__(
        self, conf: ShortlistGeneratorConfig, device: Optional[torch.device] = None
    ):
        self.conf = conf
        self.extractor = create_global_descriptor_extractor(conf, device=device)
        assert self.conf.global_desc_remove_swapped_pairs

    @torch.inference_mode()
    def __call__(
        self, scene: Scene, progress_bar: Optional[tqdm.tqdm] = None, **kwargs
    ) -> list[tuple[int, int]]:
        class_name = self.__class__.__name__

        image_paths = scene.image_paths
        if len(image_paths) <= self.conf.global_desc_fallback_threshold:
            log(
                f"[{class_name}] # of images is less than "
                f"{self.conf.global_desc_fallback_threshold}"
            )
            log(f"[{class_name}] -> Use all pairs")
            pairs = get_all_pairs(image_paths)
            scene.update_shortlist(pairs)
            return pairs

        G = self.compute_graph(scene, progress_bar=progress_bar)

        pairs_list = []
        for ci, cc in enumerate(nx.connected_components(G)):
            log(f"[{class_name}] cc({ci}): size={len(cc)}")
            if len(cc) >= 2:
                for i, j in combinations(list(cc), 2):
                    pair = tuple(sorted((int(i), int(j))))
                    pairs_list.append(pair)

        pairs_list = sorted(list(set(pairs_list)))
        log(f"[{class_name}] #pairs: {len(pairs_list)}")
        scene.update_shortlist(pairs_list)
        return pairs_list

    @torch.inference_mode()
    def compute_graph(
        self, scene: Scene, progress_bar: Optional[tqdm.tqdm] = None
    ) -> nx.Graph:
        class_name = self.__class__.__name__

        feats = extract_global_features(
            scene.image_paths,
            self.extractor,
            self.conf.global_desc_batch_size,
            num_workers=self.conf.global_desc_num_workers,
            progress_bar=progress_bar,
        )
        dists = torch.cdist(feats, feats, p=2).detach().cpu().numpy()

        ranks = np.argsort(dists)
        topk_ranks = ranks[:, : self.conf.global_desc_topk]
        topk_dists = np.take_along_axis(dists, topk_ranks, axis=1)

        xs = np.arange(len(feats))
        max_num_pairs = len(list(combinations(xs, 2)))
        log(f"[{class_name}] #pairs(max): {max_num_pairs}")

        G = nx.Graph()
        G.add_nodes_from(xs)
        for x, _ranks, _dists in zip(xs, topk_ranks, topk_dists):
            for r, d in zip(_ranks, _dists):
                if x == r:
                    continue
                if d <= self.conf.global_desc_similar_distance_threshold:
                    G.add_edge(int(x), int(r))

        if self.conf.gdcc_degree_threshold is not None:
            n = len(G.nodes())
            if n > self.conf.gdcc_degree_threshold + 1:
                removes = [
                    x
                    for x in G.nodes()
                    if G.degree(x) <= self.conf.gdcc_degree_threshold
                ]
                G.remove_nodes_from(removes)

        return G
