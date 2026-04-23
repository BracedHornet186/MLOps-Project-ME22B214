from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

import models.patchnetvlad.models.models_generic
from scripts.data import FilePath, resolve_model_path
from global_descriptors.base import DescriptorExtractor
from models.config import PatchNetVLADModelConfig
from models.patchnetvlad.model import (
    create_patch_netvlad_model,
    create_patch_netvlad_transforms,
)
from pipelines.scene import Scene


class PatchNetVLADGlobalDescriptorExtractor(DescriptorExtractor):
    def __init__(
        self, conf: PatchNetVLADModelConfig, device: torch.device | None = None
    ):
        self.conf = conf
        self.device = device
        self.model = create_patch_netvlad_model(conf, device=device)
        print(
            f"[PatchNetVLADGlobalDescriptorExtractor] "
            f"Loaded weights from {resolve_model_path(conf.weight_path)}"
        )

    @torch.inference_mode()
    def __call__(self, x: Any) -> torch.Tensor:
        assert self.conf.pooling.lower() == "patchnetvlad"
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            x = x.to(self.device)
            image_encoding = self.model.encoder(x)
            _, vlad_global = self.model.pool(image_encoding)
            vlad_global_pca = (
                models.patchnetvlad.models.models_generic.get_pca_encoding(
                    self.model, vlad_global
                )
            )

        return vlad_global_pca

    def create_dataset(self, image_paths: list[str | Path], *args, **kwargs) -> Dataset:
        return ImageDataset(image_paths, self.conf, *args, **kwargs)

    def create_dataset_from_scene(self, scene: Scene) -> Dataset:
        return ImageDataset.from_scene(scene, self)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: list[str | Path],
        conf: PatchNetVLADModelConfig,
        image_reader: Callable | None = None,
    ):
        self.image_paths = image_paths
        self.conf = conf
        self.image_reader = image_reader or cv2.imread
        self.transforms = create_patch_netvlad_transforms(
            resize=(conf.resize_height, conf.resize_width)
        )

    @classmethod
    def from_scene(
        cls, scene: Scene, extractor: PatchNetVLADGlobalDescriptorExtractor
    ) -> ImageDataset:
        return ImageDataset(
            scene.image_paths, extractor.conf, image_reader=scene.get_image
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i: int) -> Any:
        path = self.image_paths[i]
        img = self.read_image(path, reader=self.image_reader)
        x = self.image_preprocess(img)
        return x

    def read_image(self, path: FilePath, reader: Callable = cv2.imread) -> Any:
        img = reader(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img

    def image_preprocess(self, img: Any) -> Any:
        return self.transforms(img)
