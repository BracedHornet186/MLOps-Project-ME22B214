from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import tqdm

from pipelines.scene import Scene
from pipelines.verification import compute_ransac_inlier_counts
from shortlists.base import ShortlistUpdater
from shortlists.config import PreMatchingShortlistUpdaterConfig
from shortlists.imc2024 import create_global_descriptor_extractors, extract_global_features
from storage import InMemoryMatchedKeypointStorage, Line2DSegmentStorage


class PreMatchingShortlistUpdater(ShortlistUpdater):
    def __init__(
        self,
        conf: PreMatchingShortlistUpdaterConfig,
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
        **kwargs,
    ) -> List[Tuple[int, int]]:
        assert scene.shortlist
        print(f"[PreMatchingShortlistUpdater] Starts")

        if mk_storage is not None:
            assert mk_storage_list is None
            mk_storage_list = [mk_storage]
        elif mk_storage_list is not None:
            assert mk_storage is None
        else:
            raise ValueError

        base_pairs = scene.shortlist

        pairs_list: list[list[tuple[int, int]]] = []

        if self.extractors:
            assert self.conf.global_descriptors
            global_features_list = extract_global_features(
                self.extractors,
                self.conf.global_descriptors,
                scene,
                progress_bar=progress_bar
            )
            for feats, c in zip(global_features_list, self.conf.global_descriptors):
                pairs = []
                dists = torch.cdist(feats, feats, p=2).detach().cpu().numpy()
                ranks = np.argsort(dists, axis=-1)

                # NOTE:
                # When using multiple extractors,
                # ranks and dists that the last one extracts are used 
                topk_ranks, topk_dists = self.to_topk_table(feats)
                scene.update_topk_table(topk_ranks=topk_ranks, topk_dists=topk_dists)

                for i, j in np.stack(np.where(dists <= c.similar_distance_threshold)).T.tolist():
                    if i >= j:
                        continue
                    pair = tuple(sorted((int(i), int(j))))
                    pairs.append(pair)
                pairs = list(sorted(list(set(pairs))))
                pairs_list.append(pairs)

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

            pairs = []
            for i, j in base_pairs:
                key1 = Path(scene.image_paths[i]).name
                key2 = Path(scene.image_paths[j]).name
                if key1 not in scores:
                    print(
                        f"[PreMatchingShortlistUpdater] Warning! "
                        f"No key1({key1}) in scores"
                    )
                    continue
                if key2 not in scores[key1]:
                    print(
                        f"[PreMatchingShortlistUpdater] Warning! "
                        f"No key2({key2}) in scores[key1]"
                    )
                    continue

                score = scores[key1][key2]
                if score < self.conf.match_threshold:
                    continue
                pairs.append((i, j))
            pairs_list.append(pairs)

        new_pairs = []
        if len(pairs_list) == 0:
            new_pairs = []
        elif len(pairs_list) == 1:
            new_pairs = pairs_list[0]
        elif self.conf.aggregation_type == "union":
            for _pairs in pairs_list:
                new_pairs += _pairs
            new_pairs = list(sorted(list(set(new_pairs))))
        elif self.conf.aggregation_type == "intersection":
            buffers = defaultdict(int)
            for _pairs in pairs_list:
                for pair in _pairs:
                    buffers[pair] += 1
            for pair, count in buffers.items():
                if count == len(pairs_list):
                    new_pairs.append(pair)
        else:
            raise ValueError

        scene.update_shortlist(new_pairs)
        return new_pairs
