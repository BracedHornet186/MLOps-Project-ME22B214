from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import kornia
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from transformers import AutoImageProcessor, AutoModel

from data import FilePath, resolve_model_path
from global_descriptors.base import DescriptorExtractor
from models.config import HuggingFaceModelConfig
from pipelines.scene import Scene


class DINOv2GlobalDescriptorExtractor(DescriptorExtractor):
    def __init__(
        self, conf: HuggingFaceModelConfig, device: torch.device | None = None
    ):
        self.conf = conf
        self.device = device
        model = AutoModel.from_pretrained(resolve_model_path(conf.pretrained_model))
        model = model.eval().to(self.device)
        self.model = model
        self.pooling_type = (conf.options or {}).get("pooling_type", "max")
        assert self.pooling_type in ("max", "mean", "cls")
        print(f"{self.__class__.__name__}: pooling_type={self.pooling_type}")

    @torch.inference_mode()
    def __call__(self, x: Any) -> torch.Tensor:
        x = x.to(self.device)
        outputs = self.model(pixel_values=x)
        if self.pooling_type == "max":
            return F.normalize(
                outputs.last_hidden_state[:, 1:].max(dim=1)[0], dim=1, p=2
            )
        elif self.pooling_type == "mean":
            return F.normalize(outputs.last_hidden_state[:, 1:].mean(dim=1), dim=1, p=2)
        elif self.pooling_type == "cls":
            return F.normalize(outputs.last_hidden_state[:, 0], dim=1, p=2)
        else:
            raise ValueError(self.pooling_type)

    def create_dataset(self, image_paths: list[str | Path], *args, **kwargs) -> Dataset:
        return ImageDataset(image_paths, self.conf, *args, **kwargs)

    def create_dataset_from_scene(self, scene: Scene) -> Dataset:
        return ImageDataset.from_scene(scene, self)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: list[str | Path],
        conf: HuggingFaceModelConfig,
        image_reader: Callable | None = None,
    ):
        self.image_paths = image_paths
        self.conf = conf
        self.image_reader = image_reader or cv2.imread
        self.processor = AutoImageProcessor.from_pretrained(
            resolve_model_path(conf.pretrained_model)
        )

    @classmethod
    def from_scene(
        cls, scene: Scene, extractor: DINOv2GlobalDescriptorExtractor
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
        img = kornia.io.load_image(path, kornia.io.ImageLoadType.RGB32)
        return img  # Shape(3, H, W)

    def image_preprocess(self, img: Any) -> Any:
        x = self.processor(images=img, return_tensors="pt", do_rescale=False)
        return x["pixel_values"][0]  # Shape(1, 3, H, W) -> Shape(3, H, W)
