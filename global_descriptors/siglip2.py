from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data.dataset import Dataset
from transformers.image_utils import load_image
from transformers.models.siglip import SiglipImageProcessor, SiglipVisionModel

from data import FilePath, resolve_model_path
from global_descriptors.base import CustomDescriptorExtractor
from models.config import HuggingFaceModelConfig
from pipelines.scene import Scene


class SigLIP2GlobalDescriptorExtractor(CustomDescriptorExtractor):
    """google/siglip2-so400m-patch14-384"""

    def __init__(
        self, conf: HuggingFaceModelConfig, device: torch.device | None = None
    ):
        self.conf = conf
        self.device = device or torch.device("cuda")
        model = SiglipVisionModel.from_pretrained(
            resolve_model_path(conf.pretrained_model)
        )
        self.model = model.eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, x: Any) -> torch.Tensor:
        assert len(x) == 1, "Required: batch_size=1"
        x = x.to(self.device, non_blocking=True)
        outputs = self.model(pixel_values=x)
        feat = outputs.pooler_output
        return feat.to(torch.float32)

    def get_dataloader_params(self) -> dict:
        return {}

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
        self.processor = SiglipImageProcessor.from_pretrained(
            str(resolve_model_path(conf.pretrained_model))
        )

    @classmethod
    def from_scene(
        cls, scene: Scene, extractor: SigLIP2GlobalDescriptorExtractor
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
        img = reader(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img).convert("RGB")

    def image_preprocess(self, img: Any) -> Any:
        x = self.processor(images=img, return_tensors="pt")
        return x["pixel_values"][0]  # Shape(1, 3, H, W) -> Shape(3, H, W)


def identity(x: Any) -> Any:
    return x
