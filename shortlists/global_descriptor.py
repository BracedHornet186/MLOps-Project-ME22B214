from __future__ import annotations

import copy
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

from global_descriptors.apgem import APGeMGlobalDescriptorExtractor
from global_descriptors.base import (
    ConcatDescriptorExtractor,
    CustomDescriptorExtractor,
    DescriptorExtractor,
)
from global_descriptors.config import GlobalDescriptorExtractorConfigProtocol
from global_descriptors.cvnet import CVNetGlobalDescriptorExtractor
from global_descriptors.dinov2 import DINOv2GlobalDescriptorExtractor
from global_descriptors.dinov2_salad import DINOv2SALADGlobalDescriptorExtractor
from global_descriptors.isc import ISCGlobalDescriptorExtractor
from global_descriptors.mast3r_retrieval_spoc import (
    MASt3RRetrievalSPoCGlobalDescriptorExtractor,
)
from global_descriptors.mast3r_spoc import MASt3RSPoCGlobalDescriptorExtractor
from global_descriptors.moge_depth_hog import MoGeDepthHOGGlobalDescriptorExtractor
from global_descriptors.moge_dinov2 import MoGeDepthDINOv2GlobalDescriptorExtractor
from global_descriptors.moge_feature import MoGeGlobalDescriptorExtractor
from global_descriptors.patch_netvlad import PatchNetVLADGlobalDescriptorExtractor
from global_descriptors.siglip2 import SigLIP2GlobalDescriptorExtractor
from pipelines.scene import Scene
from shortlists.base import ShortlistGenerator, get_all_pairs
from shortlists.config import ShortlistGeneratorConfig
from workspace import log


class GlobalDescriptorShortlistGenerator(ShortlistGenerator):
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

        pairs_list = []
        for i, (_ranks, _dists) in enumerate(zip(topk_ranks, topk_dists)):
            mask = np.bitwise_and(
                _dists <= self.conf.global_desc_similar_distance_threshold, _ranks != i
            )
            js = _ranks[mask]
            if len(js) == 0 and self.conf.global_desc_num_refills_when_no_matches > 0:
                nearest_indices = np.argsort(_dists)[
                    1 : self.conf.global_desc_num_refills_when_no_matches + 1
                ]
                js = _ranks[nearest_indices]

            for j in js:
                if self.conf.global_desc_remove_swapped_pairs:
                    pair = tuple(sorted((int(i), int(j))))
                else:
                    pair = (int(i), int(j))
                pairs_list.append(pair)

        pairs_list = sorted(list(set(pairs_list)))
        scene.update_shortlist(pairs_list).update_topk_table(topk_ranks, topk_dists)
        return pairs_list

    def extract_global_features(
        self, scene: Scene, progress_bar: Optional[tqdm.tqdm] = None
    ) -> torch.Tensor:
        return extract_global_features(
            scene.image_paths,
            self.extractor,
            self.conf.global_desc_batch_size,
            num_workers=self.conf.global_desc_num_workers,
            progress_bar=progress_bar,
        )

    def to_topk_table(self, feats: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        dists = torch.cdist(feats, feats, p=2).detach().cpu().float().numpy()

        ranks = np.argsort(dists)
        topk_ranks = ranks[:, : self.conf.global_desc_topk]
        topk_dists = np.take_along_axis(dists, topk_ranks, axis=1)

        return topk_ranks, topk_dists


def create_global_descriptor_extractor(
    conf: GlobalDescriptorExtractorConfigProtocol, device: Optional[torch.device] = None
) -> DescriptorExtractor | ConcatDescriptorExtractor:
    assert conf.global_desc_model
    multiple_desc_names = conf.global_desc_model.split(",")
    if len(multiple_desc_names) > 1:
        extractors = []
        for i, name in enumerate(multiple_desc_names):
            name = name.rstrip()
            _conf = copy.deepcopy(conf)
            _conf.global_desc_model = name
            extractor = create_global_descriptor_extractor(_conf, device=device)
            extractors.append(extractor)
            print(f"ConcatDescriptorExtractor | [{i}]: {extractor.__class__.__name__}")
        return ConcatDescriptorExtractor(extractors)
    if conf.global_desc_model == "apgem":
        assert conf.apgem
        return APGeMGlobalDescriptorExtractor(conf.apgem, device=device)
    if conf.global_desc_model == "cvnet":
        assert conf.cvnet
        return CVNetGlobalDescriptorExtractor(conf.cvnet, device=device)
    if conf.global_desc_model == "dinov2":
        assert conf.dinov2
        return DINOv2GlobalDescriptorExtractor(conf.dinov2, device=device)
    if conf.global_desc_model == "dinov2_salad":
        assert conf.dinov2_salad
        return DINOv2SALADGlobalDescriptorExtractor(conf.dinov2_salad, device=device)
    if conf.global_desc_model == "patchnetvlad":
        assert conf.patchnetvlad
        return PatchNetVLADGlobalDescriptorExtractor(conf.patchnetvlad, device=device)
    if conf.global_desc_model == "mast3r_spoc":
        assert conf.mast3r_spoc
        return MASt3RSPoCGlobalDescriptorExtractor(conf.mast3r_spoc, device=device)
    if conf.global_desc_model == "mast3r_retrieval_spoc":
        assert conf.mast3r_retrieval_spoc
        return MASt3RRetrievalSPoCGlobalDescriptorExtractor(
            conf.mast3r_retrieval_spoc, device=device
        )
    if conf.global_desc_model == "moge":
        assert conf.moge
        return MoGeGlobalDescriptorExtractor(conf.moge, device=device)
    if conf.global_desc_model == "moge_depth_hog":
        assert conf.moge_depth_hog
        return MoGeDepthHOGGlobalDescriptorExtractor(conf.moge_depth_hog, device=device)
    if conf.global_desc_model == "moge_dinov2":
        assert conf.moge_dinov2
        assert conf.dinov2
        dinov2_extractor = DINOv2GlobalDescriptorExtractor(conf.dinov2, device=device)
        return MoGeDepthDINOv2GlobalDescriptorExtractor(
            conf.moge_dinov2,
            dinov2_extractor,
            device=device,
        )
    if conf.global_desc_model == "siglip2":
        assert conf.siglip2
        return SigLIP2GlobalDescriptorExtractor(conf.siglip2, device=device)
    if conf.global_desc_model == "isc":
        assert conf.isc
        return ISCGlobalDescriptorExtractor(conf.isc, device=device)
    raise ValueError


@torch.inference_mode()
def extract_global_features(
    image_paths: Sequence[str | Path],
    extractor: DescriptorExtractor | ConcatDescriptorExtractor,
    batch_size: int,
    num_workers: int = 0,
    progress_bar: tqdm.tqdm | None = None,
) -> torch.Tensor:
    if isinstance(extractor, ConcatDescriptorExtractor):
        feats_list = []
        for _extractor in extractor.extractors:
            feats = extract_global_features(
                image_paths,
                _extractor,
                batch_size,
                num_workers=num_workers,
                progress_bar=progress_bar,
            )
            feats_list.append(feats)
        feats = extractor.concat_features(feats_list)
        return feats

    dataset = extractor.create_dataset(list(image_paths))
    params: dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    params.update(extractor.get_dataloader_params())
    loader = torch.utils.data.DataLoader(
        dataset,
        **params,
    )
    device = extractor.device or torch.device("cuda")

    feats = []
    for i, x in enumerate(loader, start=1):
        if isinstance(extractor, CustomDescriptorExtractor):
            with torch.autocast(device.type):
                f = extractor(x)
        else:
            if extractor.device:
                x = x.to(device, non_blocking=True)
            with torch.autocast(device.type):
                f = extractor(x)
        feats.append(f)

        if progress_bar:
            progress_bar.set_postfix_str(
                f"Global descriptors extraction ({i}/{len(loader)})"
            )

    feats = torch.cat(feats)
    del dataset
    del loader
    return feats
