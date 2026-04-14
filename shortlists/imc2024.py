from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
import tqdm

from pipelines.scene import Scene
from shortlists.base import ShortlistGenerator, get_all_pairs
from shortlists.config import GlobalDescriptorConfig, IMC2024ShortlistGeneratorConfig
from shortlists.global_descriptor import (
    APGeMGlobalDescriptorExtractor,
    CVNetGlobalDescriptorExtractor,
    DescriptorExtractor,
    DINOv2GlobalDescriptorExtractor,
    DINOv2SALADGlobalDescriptorExtractor,
    MASt3RRetrievalSPoCGlobalDescriptorExtractor,
    PatchNetVLADGlobalDescriptorExtractor,
)
from workspace import log


class IMC2024ShortlistGenerator(ShortlistGenerator):
    def __init__(
        self,
        conf: IMC2024ShortlistGeneratorConfig,
        device: Optional[torch.device] = None,
    ):
        self.conf = conf
        self.extractors = create_global_descriptor_extractors(
            conf.global_descriptors, device=device
        )
        print(f"[IMC2024ShortlistGenerator] {self.extractors}")

    @torch.inference_mode()
    def __call__(
        self, scene: Scene, progress_bar: Optional[tqdm.tqdm] = None, **kwargs
    ) -> list[tuple[int, int]]:
        assert self.conf.global_descriptors

        image_paths = scene.image_paths
        if len(image_paths) <= self.conf.all_pairs_fallback_threshold:
            # Fallback to all_pairs
            log(f"# of images is less than {self.conf.all_pairs_fallback_threshold}")
            log("-> Use all pairs")

            pairs = get_all_pairs(image_paths)
            topk_ranks, topk_dists = None, None
            if self.conf.compute_feats_if_fallback:
                # Fallback, but compute features and topk tables
                multiple_model_features = extract_global_features(
                    self.extractors,
                    self.conf.global_descriptors,
                    scene,
                    progress_bar=progress_bar,
                )
                if self.conf.multiple_descs_fusion_type == "top-ranking":
                    raise NotImplementedError
                feats = fuse_multiple_features(multiple_model_features, conf=self.conf)
                topk_ranks, topk_dists = to_topk_table(feats, self.conf.topk)
            scene.update_shortlist(pairs).update_topk_table(topk_ranks, topk_dists)
            return pairs

        multiple_model_features = extract_global_features(
            self.extractors,
            self.conf.global_descriptors,
            scene,
            progress_bar=progress_bar,
        )

        if self.conf.multiple_descs_fusion_type in ("concat", "concat-and-normalize"):
            pairs_list, topk_ranks, topk_dists = create_pairs(
                multiple_model_features, self.conf
            )
            scene.update_shortlist(pairs_list).update_topk_table(topk_ranks, topk_dists)
        else:
            raise NotImplementedError

        return pairs_list


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, scene: Scene, extractor: DescriptorExtractor):
        self.scene = scene
        self.image_paths = scene.image_paths
        self.extractor = extractor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i: int) -> Any:
        path = self.image_paths[i]
        img = self.extractor.read_image(path, reader=self.scene.get_image)
        x = self.extractor.image_preprocess(img)
        return x


def create_global_descriptor_extractors(
    global_descriptor_configs: Optional[list[GlobalDescriptorConfig]] = None,
    device: Optional[torch.device] = None,
) -> list[DescriptorExtractor]:
    extractors = []
    if not global_descriptor_configs:
        return extractors

    for c in global_descriptor_configs:
        if c.type == "apgem":
            assert c.apgem
            extractor = APGeMGlobalDescriptorExtractor(c.apgem, device=device)
        elif c.type == "cvnet":
            assert c.cvnet
            extractor = CVNetGlobalDescriptorExtractor(c.cvnet, device=device)
        elif c.type == "dinov2":
            assert c.dinov2
            extractor = DINOv2GlobalDescriptorExtractor(c.dinov2, device=device)
        elif c.type == "dinov2_salad":
            assert c.dinov2_salad
            extractor = DINOv2SALADGlobalDescriptorExtractor(
                c.dinov2_salad, device=device
            )
        elif c.type == "patchnetvlad":
            assert c.patchnetvlad
            extractor = PatchNetVLADGlobalDescriptorExtractor(
                c.patchnetvlad, device=device
            )
        elif c.type == "mast3r_retrieval_spoc":
            assert c.mast3r_retrieval_spoc
            extractor = MASt3RRetrievalSPoCGlobalDescriptorExtractor(
                c.mast3r_retrieval_spoc, device=device
            )
        else:
            raise ValueError
        extractors.append(extractor)
    return extractors


def extract_global_features(
    extractors: list[DescriptorExtractor],
    global_descriptor_configs: list[GlobalDescriptorConfig],
    scene: Scene,
    to_half: bool = False,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> list[torch.Tensor]:
    assert len(global_descriptor_configs) == len(extractors)

    # NOTE:
    # PatchNetVLAD does not support deterministic=True
    is_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)

    multiple_model_features: list[torch.Tensor] = []
    for j, c, extractor in zip(
        range(len(extractors)), global_descriptor_configs, extractors
    ):
        dataset = ImageDataset(scene, extractor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=c.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=c.num_workers,
        )

        _feats = []
        for i, x in enumerate(loader, start=1):
            if extractor.device:
                x = x.to(extractor.device, non_blocking=True)
            f = extractor(x)
            if to_half:
                f = f.half()
            _feats.append(f)

            if progress_bar:
                progress_bar.set_postfix_str(
                    f"Descriptors[{j}] extracts ({i}/{len(loader)})"
                )

        feats = torch.cat(_feats)
        multiple_model_features.append(feats)

        del dataset
        del loader

    torch.use_deterministic_algorithms(is_deterministic)

    return multiple_model_features


def fuse_multiple_features(
    multiple_model_features: list[torch.Tensor], conf: IMC2024ShortlistGeneratorConfig
) -> torch.Tensor:
    if conf.multiple_descs_fusion_type == "concat":
        feats = torch.cat(multiple_model_features, axis=1)  # type: ignore
    elif conf.multiple_descs_fusion_type == "concat-and-normalize":
        feats = torch.cat(multiple_model_features, axis=1)  # type: ignore
        feats = torch.nn.functional.normalize(feats)
        print(feats.shape)
    else:
        raise ValueError
    return feats


def to_topk_table(feats: torch.Tensor, topk: int) -> tuple[np.ndarray, np.ndarray]:
    dists = torch.cdist(feats, feats, p=2).detach().cpu().float().numpy()

    ranks = np.argsort(dists)
    topk_ranks = ranks[:, :topk]
    topk_dists = np.take_along_axis(dists, topk_ranks, axis=1)

    return topk_ranks, topk_dists


def create_pairs(
    multiple_model_features: list[torch.Tensor], conf: IMC2024ShortlistGeneratorConfig
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    feats = fuse_multiple_features(multiple_model_features, conf)
    topk_ranks, topk_dists = to_topk_table(feats, conf.topk)

    pairs_list = []
    for i, (_ranks, _dists) in enumerate(zip(topk_ranks, topk_dists)):
        if conf.similar_distance_threshold is None:
            mask = _ranks != i
        else:
            mask = np.bitwise_and(
                _dists <= conf.similar_distance_threshold, _ranks != i
            )
        js = _ranks[mask]
        if len(js) == 0 and conf.num_refills_when_no_matches > 0:
            nearest_indices = np.argsort(_dists)[
                1 : conf.num_refills_when_no_matches + 1
            ]
            js = _ranks[nearest_indices]

        for j in js:
            if conf.remove_swapped_pairs:
                pair = tuple(sorted((int(i), int(j))))
            else:
                pair = (int(i), int(j))
            pairs_list.append(pair)

    pairs_list = sorted(list(set(pairs_list)))
    return pairs_list, topk_ranks, topk_dists
