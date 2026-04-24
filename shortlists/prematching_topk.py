from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
import tqdm

from pipelines.scene import Scene
from pipelines.verification import compute_ransac_inlier_counts
from shortlists.base import ShortlistUpdater
from shortlists.config import PreMatchingTopKShortlistUpdaterConfig
from shortlists.imc2024 import (
    create_global_descriptor_extractors,
    extract_global_features,
)
from scripts.storage import (
    InMemoryMatchedKeypointStorage,
    InMemoryMatchingStorage,
    Line2DSegmentStorage,
)


class PreMatchingTopKShortlistUpdater(ShortlistUpdater):
    def __init__(
        self,
        conf: PreMatchingTopKShortlistUpdaterConfig,
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
        print("[PreMatchingTopKShortlistUpdater] Starts")

        if mk_storage is not None:
            assert mk_storage_list is None
            mk_storage_list = [mk_storage]
        elif mk_storage_list is not None:
            assert mk_storage is None
        else:
            raise ValueError

        print("[PreMatchingTopKShortlistUpdater] Use following data")
        print(f"  - {len(mk_storage_list)} local features")
        print(f"  - {len(self.extractors)} global features")
        if line2d_matching_storage:
            print("  - 1 line2d features")
        else:
            print("  - 0 line2d features")

        base_pairs = scene.shortlist

        pairs: set[tuple] = set()

        if self.extractors:
            assert self.conf.global_descriptors
            global_features_list = extract_global_features(
                self.extractors,
                self.conf.global_descriptors,
                scene,
                progress_bar=progress_bar,
            )
            for feats, c in zip(global_features_list, self.conf.global_descriptors):
                #dists = torch.cdist(feats, feats, p=2).detach().cpu().numpy()
                #ranks = np.argsort(dists, axis=-1)

                topk_ranks, topk_dists = self.to_topk_table(feats)

                # NOTE:
                # When using multiple extractors,
                # ranks and dists that the last one extracts are used
                scene.update_topk_table(topk_ranks=topk_ranks, topk_dists=topk_dists)

                for i, (_ranks, _dists) in enumerate(zip(topk_ranks, topk_dists)):
                    js = _ranks[1: (self.conf.topk + 1)]
                    for j in js:
                        pair = tuple(sorted((int(i), int(j))))
                        if pair in base_pairs:
                            pairs.add(pair)
                        else:
                            print(f"Warning! shortlist does not have all pairs")

        if self.conf.fallback_threshold is not None:
            if len(scene.image_paths) <= self.conf.fallback_threshold:
                print(
                    "[PreMatchingTopKShortlistUpdater] "
                    "Fallback to all pairs"
                )
                return scene.shortlist

        print(
            f"[PreMatchingTopKShortlistUpdater] "
            f"Selection based on global descriptors finished. "
            f"Totally {len(pairs)} pairs have been added"
        )

        for mk_storage in mk_storage_list:
            k_storage, m_storage = mk_storage.to_keypoints_and_matches()

            if self.conf.ransac:
                scores = compute_ransac_inlier_counts(
                    k_storage, m_storage, self.conf.ransac, progress_bar=progress_bar
                )
            else:
                m_storage = m_storage.to_memory()
                scores = {}
                for key1, group in m_storage.matches.items():
                    if key1 not in scores:
                        scores[key1] = {}
                    for key2, idxs in group.items():
                        scores[key1][key2] = int(len(idxs))

            for key1 in scores.keys():
                key2list = list(scores[key1].keys())
                scorelist = np.array(list(scores[key1].values()))
                idx = np.argsort(-scorelist)[: self.conf.topk]
                topk_key2_list = [key2list[p] for p in idx]
                for key2 in topk_key2_list:
                    i = scene.short_key_to_idx(key1)
                    j = scene.short_key_to_idx(key2)
                    pair = tuple(sorted((int(i), int(j))))
                    if pair in base_pairs:
                        pairs.add(pair)
                    else:
                        print(f"Warning! shortlist does not have all pairs")
        print(
            f"[PreMatchingTopKShortlistUpdater] "
            f"Selection based on local feature matching finished. "
            f"Totally {len(pairs)} pairs have been selected"
        )

        if line2d_matching_storage:
            scores = {}
            for key1, group in line2d_matching_storage.matches.items():
                if key1 not in scores:
                    scores[key1] = {}
                for key2, idxs in group.items():
                    scores[key1][key2] = int(len(idxs))

            for key1 in scores.keys():
                key2list = list(scores[key1].keys())
                scorelist = np.array(list(scores[key1].values()))
                idx = np.argsort(-scorelist)[: self.conf.topk]
                topk_key2_list = [key2list[p] for p in idx]
                for key2 in topk_key2_list:
                    i = scene.short_key_to_idx(key1)
                    j = scene.short_key_to_idx(key2)
                    pair = tuple(sorted((int(i), int(j))))
                    if pair in base_pairs:
                        pairs.add(pair)
                    else:
                        print(f"Warning! shortlist does not have all pairs")
            print(
                f"[PreMatchingTopKShortlistUpdater] "
                f"Selection based on line2d feature matching finished. "
                f"Totally {len(pairs)} pairs have been selected"
            )

        new_pairs = list(sorted(list(pairs)))
        scene.update_shortlist(new_pairs)

        if self.conf.path_length_for_additional_pairs is not None:
            current_pairs = set(new_pairs)
            additional_pairs: list[tuple[int, int]] = []

            max_length = self.conf.path_length_for_additional_pairs
            assert max_length >= 2
            scene_graph = scene.make_scene_graph()
            G = scene_graph["graph"]
            all_nodes = list(range(len(scene.image_paths)))

            degree_dict = dict(nx.degree(G))
            print(f"degree: {degree_dict}")
            for i in all_nodes:
                degree = degree_dict.get(i)
                if degree is None:
                    continue
                if degree >= self.conf.satisfied_edges_threshold:
                    continue
                shortest_length_dict = nx.single_source_dijkstra_path_length(G, source=i)
                print(f"[{i}] Shortest length dict: {shortest_length_dict}")
                for j, length_ij in shortest_length_dict.items():
                    if i == j:
                        continue
                    if length_ij > max_length:
                        continue
                    pair = tuple(sorted((int(i), int(j))))
                    if pair in current_pairs:
                        continue
                    additional_pairs.append(pair)   # type: ignore
                    print(f"[{i}] Added new pair: {pair} (length={length_ij})")
            
            new_pairs = [pair for pair in new_pairs] + additional_pairs
            new_pairs = list(sorted(new_pairs))

            print(
                f"[PreMatchingTopKShortlistUpdater] "
                f"Appending pairs based on path length in a scene graph finished. "
                f"Totally {len(new_pairs)} pairs have been selected"
            )
            new_pairs = list(sorted(list(new_pairs)))
            scene.update_shortlist(new_pairs)

        return new_pairs
