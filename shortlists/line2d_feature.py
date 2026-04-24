from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import tqdm

from pipelines.scene import Scene
from pipelines.verification import compute_ransac_inlier_counts
from shortlists.base import ShortlistUpdater
from shortlists.config import Line2DFeatureShortlistUpdaterConfig
from shortlists.imc2024 import create_global_descriptor_extractors, extract_global_features
from scripts.storage import InMemoryMatchedKeypointStorage, Line2DFeatureStorage
from utils.vlad import VLAD


class Line2DFeatureShortlistUpdater(ShortlistUpdater):
    def __init__(
        self,
        conf: Line2DFeatureShortlistUpdaterConfig,
        device: Optional[torch.device] = None,
    ):
        self.conf = conf
        self.device = device

    @torch.inference_mode()
    def __call__(
        self,
        scene: Scene,
        progress_bar: Optional[tqdm.tqdm] = None,
        mk_storage: Optional[InMemoryMatchedKeypointStorage] = None,
        mk_storage_list: Optional[list[InMemoryMatchedKeypointStorage]] = None,
        line2d_feature_storage: Optional[Line2DFeatureStorage] = None,
        **kwargs,
    ) -> List[Tuple[int, int]]:
        assert scene.shortlist
        print("[Line2DFeatureShortlistUpdater] Starts")

        base_pairs = scene.shortlist

        pairs_list: list[list[tuple[int, int]]] = []

        assert line2d_feature_storage is not None
        all_feats = []
        for path in scene.image_paths:
            _, descinfo = line2d_feature_storage.get(path)
            # DeepLSD, SOLD2
            feats, _ = descinfo
            feats = feats.T
            all_feats.append(feats)

        print(f"Computing VLAD feats")
        vlad_feats = VLAD(k=8).fit_transform(all_feats)
        print(f"VLAD feats: {vlad_feats.shape}")

        x = torch.from_numpy(vlad_feats).cuda()
        dists = torch.cdist(x, x).cpu().numpy()

        raise NotImplementedError
        for i, _dists in enumerate(dists):
            pass

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
