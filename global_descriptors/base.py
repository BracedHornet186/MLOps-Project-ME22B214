from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, Self

import torch
import torch.nn.functional as F
import torch.utils.data

from pipelines.scene import Scene


class DescriptorExtractor:
    device: torch.device | None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def create_dataset(
        self, image_paths: list[str | Path], *args, **kwargs
    ) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def create_dataset_from_scene(self, scene: Scene) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def get_dataloader_params(self) -> dict:
        return {}


class CustomDescriptorExtractor(DescriptorExtractor):
    def __call__(self, inputs: Any) -> torch.Tensor:
        raise NotImplementedError


class ConcatDescriptorExtractor:
    def __init__(
        self,
        extractors: Sequence[DescriptorExtractor],
        normalize_part: bool = True,
        normalize_concat: bool = True,
    ):
        self.extractors = list(extractors)
        self.normalize_part = normalize_part
        self.normalize_concat = normalize_concat

    def concat_features(self, feats_list: list[torch.Tensor]) -> torch.Tensor:
        if self.normalize_part:
            feats_list = [F.normalize(feats) for feats in feats_list]

        feats = torch.concat(feats_list, dim=1)
        if self.normalize_concat:
            feats = F.normalize(feats)

        return feats
