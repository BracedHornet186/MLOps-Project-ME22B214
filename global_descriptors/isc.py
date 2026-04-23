from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from scripts.data import FilePath, resolve_model_path
from global_descriptors.base import CustomDescriptorExtractor
from models.config import ISCModelConfig
from models.isc.model import create_model, create_preprocessor
from pipelines.scene import Scene


class ISCGlobalDescriptorExtractor(CustomDescriptorExtractor):
    def __init__(self, conf: ISCModelConfig, device: torch.device | None = None):
        self.conf = conf
        self.device = device or torch.device("cuda")
        model, _, input_size, mean, std = create_model(
            resolve_model_path(conf.weight_path), self.device
        )
        self.model = model
        self.input_size = input_size
        self.mean = mean
        self.std = std

    @torch.inference_mode()
    def __call__(self, x: Any) -> torch.Tensor:
        assert len(x) == 1, "Required: batch_size=1"
        assert isinstance(x, torch.Tensor)
        x = x.to(self.device, non_blocking=True)
        feat = self.model(x)
        return feat  # (1, 256)

    def get_dataloader_params(self) -> dict:
        return {}

    def create_dataset(self, image_paths: list[str | Path], *args, **kwargs) -> Dataset:
        return ImageDataset(
            image_paths,
            self.input_size,
            copy.deepcopy(self.mean),
            copy.deepcopy(self.std),
        )

    def create_dataset_from_scene(self, scene: Scene) -> Dataset:
        return ImageDataset.from_scene(scene, self)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: list[str | Path],
        input_size: int,
        mean: Any,
        std: Any,
    ):
        self.image_paths = image_paths
        self.preprocessor = create_preprocessor(input_size, mean, std)

    @classmethod
    def from_scene(
        cls, scene: Scene, extractor: ISCGlobalDescriptorExtractor
    ) -> ImageDataset:
        return ImageDataset(
            scene.image_paths,
            extractor.input_size,
            copy.deepcopy(extractor.mean),
            copy.deepcopy(extractor.std),
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i: int) -> Any:
        path = self.image_paths[i]
        img = Image.open(path).convert("RGB")
        x = self.preprocessor(img)
        return x

    def image_preprocess(self, img: Any) -> Any:
        raise NotImplementedError


def identity(x: Any) -> Any:
    return x
