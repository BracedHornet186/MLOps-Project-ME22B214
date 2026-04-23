from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import torch
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset

from scripts.data import FilePath, resolve_model_path
from global_descriptors.base import DescriptorExtractor
from models.config import CVNetConfig
from models.cvnet.model.CVNet_Rerank_model import CVNet_Rerank
from pipelines.scene import Scene


class CVNetGlobalDescriptorExtractor(DescriptorExtractor):
    def __init__(self, conf: CVNetConfig, device: torch.device | None = None):
        self.conf = conf
        self.device = device
        model = CVNet_Rerank(conf.depth, conf.reduction_dim)
        state_dict = torch.load(
            resolve_model_path(conf.weight_path), map_location="cpu"
        )["model_state"]
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items()}
        weight_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and model_dict[k].size() == v.size()
        }

        if len(weight_dict) != len(state_dict):
            raise AssertionError("The model is not fully loaded.")

        model_dict.update(weight_dict)
        model.load_state_dict(model_dict)
        self.model = model.eval().to(device)

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: multiscale
        f = self.model.extract_global_descriptor(x)
        return f

    def create_dataset(self, image_paths: list[str | Path], *args, **kwargs) -> Dataset:
        return ImageDataset(image_paths, self.conf, *args, **kwargs)

    def create_dataset_from_scene(self, scene: Scene) -> Dataset:
        return ImageDataset.from_scene(scene, self)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: list[str | Path],
        conf: CVNetConfig,
        image_reader: Callable | None = None,
    ):
        self.image_paths = image_paths
        self.conf = conf
        self.image_reader = image_reader or cv2.imread
        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),
            ]
        )

    @classmethod
    def from_scene(
        cls, scene: Scene, extractor: CVNetGlobalDescriptorExtractor
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
        # BGR, but ok
        img = reader(str(path))
        return img

    def image_preprocess(self, img: Any) -> Any:
        return self.transforms(img)
