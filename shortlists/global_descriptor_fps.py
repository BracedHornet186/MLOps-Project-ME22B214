from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as T
import tqdm
from PIL import Image

from global_descriptors.base import CustomDescriptorExtractor, DescriptorExtractor
from pipelines.scene import Scene
from shortlists.base import ShortlistGenerator, get_all_pairs
from shortlists.config import ShortlistGeneratorConfig
from shortlists.global_descriptor import create_global_descriptor_extractor
from scripts.workspace import log


class GlobalDescriptorFPSShortlistGenerator(ShortlistGenerator):
    def __init__(
        self, conf: ShortlistGeneratorConfig, device: Optional[torch.device] = None
    ):
        self.conf = conf
        self.extractor = create_global_descriptor_extractor(conf, device=device)

    @torch.inference_mode()
    def __call__(
        self, scene: Scene, progress_bar: Optional[tqdm.tqdm] = None, **kwargs
    ) -> list[tuple[int, int]]:
        image_paths = scene.image_paths
        if len(image_paths) <= self.conf.global_desc_fallback_threshold:
            # Fallback to all_pairs
            log(f"# of images is less than {self.conf.global_desc_fallback_threshold}")
            log("-> Use all pairs")

            pairs = get_all_pairs(image_paths)
            topk_ranks, topk_dists = None, None
            if self.conf.global_desc_compute_feats_if_fallback:
                # Fallback, but compute features and topk tables
                feats = self.extract_global_features(scene, progress_bar=progress_bar)
                topk_ranks, topk_dists = self.to_topk_table(feats)
            scene.update_shortlist(pairs).update_topk_table(topk_ranks, topk_dists)
            return pairs

        feats = self.extract_global_features(scene, progress_bar=progress_bar)
        topk_ranks, topk_dists = self.to_topk_table(feats)

        dists = torch.cdist(feats, feats, p=2)
        _dists = dists.cpu().numpy()
        _sims = -_dists
        _sim_sums = _sims.sum(axis=1)
        most_common_id = int(np.argmax(_sim_sums))

        assert self.conf.global_desc_fps_n is not None
        assert self.conf.global_desc_fps_k is not None
        fps_indices, fps_dists = farthest_point_sampling(
            _dists,
            most_common_id,
            N=self.conf.global_desc_fps_n,
            dist_thresh=self.conf.global_desc_fps_dist_threshold,
        )

        pairs = set()

        # 1. Complete graph between key images
        for i in range(len(fps_indices)):
            for j in range(i + 1, len(fps_indices)):
                idx_i, idx_j = fps_indices[i], fps_indices[j]
                pairs.add((idx_i, idx_j))

        # 2. Connect non-key images to the nearest key image
        keyimg_dist_mat = _dists[:, fps_indices]
        for i in range(keyimg_dist_mat.shape[0]):
            if i in fps_indices:
                continue
            j = keyimg_dist_mat[i].argmin()
            i1, i2 = min(i, int(fps_indices[j])), max(i, int(fps_indices[j]))
            if i1 != i2 and (i1, i2) not in pairs:
                pairs.add((i1, i2))

        # 3. Add some local connections (k-NN) for each view
        if self.conf.global_desc_fps_k > 0:
            for i in range(_dists.shape[0]):
                idx = _dists[i].argsort()[: self.conf.global_desc_fps_k]
                for j in idx:
                    i1, i2 = min(i, j), max(i, j)
                    if i1 != i2 and (i1, i2) not in pairs:
                        pairs.add((i1, i2))

        pairs = [tuple(sorted([i, j])) for (i, j) in list(pairs)]
        pairs_list = sorted(list(set(pairs)))
        scene.update_shortlist(pairs_list).update_topk_table(topk_ranks, topk_dists)
        return pairs_list

    def extract_global_features(
        self, scene: Scene, progress_bar: Optional[tqdm.tqdm] = None
    ) -> torch.Tensor:
        dataset = self.extractor.create_dataset_from_scene(scene)
        params: dict[str, Any] = dict(
            batch_size=self.conf.global_desc_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.conf.global_desc_num_workers,
        )
        params.update(self.extractor.get_dataloader_params())
        loader = torch.utils.data.DataLoader(
            dataset,
            **params,
        )

        feats = []
        for i, x in enumerate(loader, start=1):
            if isinstance(self.extractor, CustomDescriptorExtractor):
                with torch.autocast(self.extractor.device.type):
                    f = self.extractor(x)
            else:
                if self.extractor.device:
                    x = x.to(self.extractor.device, non_blocking=True)
                with torch.autocast(self.extractor.device.type):
                    f = self.extractor(x)
            feats.append(f)

            if progress_bar:
                progress_bar.set_postfix_str(
                    f"Global descriptors extraction ({i}/{len(loader)})"
                )

        feats = torch.cat(feats)
        del dataset
        del loader
        return feats

    def to_topk_table(self, feats: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        dists = torch.cdist(feats, feats, p=2).detach().cpu().float().numpy()

        ranks = np.argsort(dists)
        topk_ranks = ranks[:, : self.conf.global_desc_topk]
        topk_dists = np.take_along_axis(dists, topk_ranks, axis=1)

        return topk_ranks, topk_dists


def farthest_point_sampling(
    dist: np.ndarray,
    initial_point: int,
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
