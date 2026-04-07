from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from moge.model.v1 import MoGeModel
from torch.utils.data.dataset import Dataset

from data import FilePath, resolve_model_path
from global_descriptors.base import CustomDescriptorExtractor
from models.config import MoGeModelConfig
from models.moge.feature import MoGeFeatureModel
from pipelines.scene import Scene


class MoGeGlobalDescriptorExtractor(CustomDescriptorExtractor):
    def __init__(self, conf: MoGeModelConfig, device: torch.device | None = None):
        self.conf = conf
        self.moge_feature_model = MoGeFeatureModel(
            MoGeModel.from_pretrained("Ruicheng/moge-vitl"),
            device,
        )
        self.device = device or torch.device("cuda")

    @torch.inference_mode()
    def __call__(self, x: Any) -> torch.Tensor:
        assert len(x) == 1, "Required: batch_size=1"
        x = x.to(self.device, non_blocking=True)
        feat = self.moge_feature_model(x)  # Shape(B=1, 512)
        return feat

    def get_dataloader_params(self) -> dict:
        return {}

    def create_dataset(self, image_paths: list[str | Path], *args, **kwargs) -> Dataset:
        return ImageDataset(image_paths)

    def create_dataset_from_scene(self, scene: Scene) -> Dataset:
        return ImageDataset.from_scene(scene, self)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: list[str | Path],
    ):
        self.image_paths = image_paths

    @classmethod
    def from_scene(
        cls, scene: Scene, extractor: MoGeGlobalDescriptorExtractor
    ) -> ImageDataset:
        return ImageDataset(scene.image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i: int) -> Any:
        path = self.image_paths[i]
        input_image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        input_image = torch.tensor(input_image / 255, dtype=torch.float32).permute(
            2, 0, 1
        )
        return input_image

    def image_preprocess(self, img: Any) -> Any:
        raise NotImplementedError


def identity(x: Any) -> Any:
    return x
