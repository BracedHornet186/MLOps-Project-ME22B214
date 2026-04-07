from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset

import models.dirtorch.model
import models.dirtorch.utils.common
from data import FilePath, resolve_model_path
from global_descriptors.base import DescriptorExtractor
from models.config import APGeMConfig
from pipelines.scene import Scene


class APGeMGlobalDescriptorExtractor(DescriptorExtractor):
    def __init__(self, conf: APGeMConfig, device: torch.device | None = None):
        self.conf = conf
        self.device = device
        self.model = models.dirtorch.model.load_model(
            resolve_model_path(conf.weight_path), device=device
        )
        assert self.conf.whiten_name in self.model.pca

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.model.preprocess["mean"]
        std = self.model.preprocess["std"]
        x = x - x.new_tensor(mean)[:, None, None]
        x = x / x.new_tensor(std)[:, None, None]

        desc = self.model(x)
        desc = desc.unsqueeze(0)  # batch dimension
        if self.conf.whiten_name:
            pca = self.model.pca[self.conf.whiten_name]
            desc = models.dirtorch.model.common.whiten_features(
                desc.cpu().numpy(), pca, **self.conf.get_whiten_params()
            )
            desc = torch.from_numpy(desc)
        return desc

    def create_dataset(self, image_paths: list[str | Path], *args, **kwargs) -> Dataset:
        return ImageDataset(image_paths, self.conf, *args, **kwargs)

    def create_dataset_from_scene(self, scene: Scene) -> Dataset:
        return ImageDataset.from_scene(scene, self)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: list[str | Path],
        conf: APGeMConfig,
        image_reader: Callable | None = None,
    ):
        self.image_paths = image_paths
        self.conf = conf
        self.image_reader = image_reader or cv2.imread

    @classmethod
    def from_scene(
        cls, scene: Scene, extractor: APGeMGlobalDescriptorExtractor
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

    def image_preprocess(self, img: Any) -> Any:
        assert isinstance(img, np.ndarray)
        size = img.shape[:2][::-1]
        if self.conf.resize_max and max(size) > self.conf.resize_max:
            scale = self.conf.resize_max / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            interp = cv2.INTER_AREA
            h, w = img.shape[:2]
            if interp == cv2.INTER_AREA and (w < size_new[0] or h < size_new[1]):
                interp = cv2.INTER_LINEAR
            resized_img = cv2.resize(img, size_new, interpolation=interp)
        else:
            resized_img = img

        image = resized_img.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.0
        return image

    def read_image(self, path: FilePath, reader: Callable = cv2.imread) -> Any:
        img = reader(str(path))
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img
