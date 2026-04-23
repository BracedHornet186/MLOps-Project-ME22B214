from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from dust3r.utils.image import load_images
from mast3r.model import AsymmetricMASt3R
from mast3r.retrieval.processor import Retriever
from torch.utils.data.dataset import Dataset

from scripts.data import FilePath, resolve_model_path
from global_descriptors.base import CustomDescriptorExtractor
from models.config import MASt3RRetrievalModelConfig
from models.mast3r.model import get_mast3r_model
from pipelines.scene import Scene


class MASt3RRetrievalSPoCGlobalDescriptorExtractor(CustomDescriptorExtractor):
    def __init__(
        self, conf: MASt3RRetrievalModelConfig, device: torch.device | None = None
    ):
        self.conf = conf
        self.device = device or torch.device("cuda")
        backbone = get_mast3r_model(
            resolve_model_path(conf.mast3r.weight_path), self.device
        )
        assert conf.mast3r.retrieval_weight_path and conf.mast3r.retrieval_codebook_path
        self.retriever = Retriever(
            str(resolve_model_path(conf.mast3r.retrieval_weight_path)),
            backbone=backbone,
            device=self.device,  # type: ignore
        )
        print(
            f"[{self.__class__.__name__}] "
            f"weight={resolve_model_path(conf.mast3r.weight_path)}, "
            f"retrieval_weight={resolve_model_path(conf.mast3r.retrieval_weight_path)}, "
            f"retrieval_codebook={resolve_model_path(conf.mast3r.retrieval_codebook_path)}, "
        )

    def __call__(self, inputs: Any) -> torch.Tensor:
        assert len(inputs) == 1, "Required: batch_size=1"
        sample = inputs[0]
        assert isinstance(sample, dict)

        sample["img"] = sample["img"].to(self.device, non_blocking=True)
        feat = self.retriever.model.forward_global(sample)  # Shape(B=1, 1024)
        return feat

    def get_dataloader_params(self) -> dict:
        return {
            "collate_fn": identity,
        }

    def create_dataset(self, image_paths: list[str | Path], *args, **kwargs) -> Dataset:
        return ImageDataset(image_paths, self.conf, *args, **kwargs)

    def create_dataset_from_scene(self, scene: Scene) -> Dataset:
        return ImageDataset.from_scene(scene, self)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: list[str | Path],
        conf: MASt3RRetrievalModelConfig,
        image_reader: Callable | None = None,
    ):
        self.image_paths = image_paths
        self.conf = conf
        self.image_reader = image_reader or cv2.imread
        self.imsize = 512

    @classmethod
    def from_scene(
        cls, scene: Scene, extractor: MASt3RRetrievalSPoCGlobalDescriptorExtractor
    ) -> ImageDataset:
        return ImageDataset(
            scene.image_paths, extractor.conf, image_reader=scene.get_image
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i: int) -> Any:
        path = self.image_paths[i]
        sample: dict = self.read_image(path, reader=self.image_reader)
        # NOTE:
        # Use a dataloader with batch_size=1 and collate_fn=lambda x: x
        return sample

    def read_image(self, path: FilePath, reader: Callable = cv2.imread) -> Any:
        sample = load_images([path], size=self.imsize, verbose=False)[0]
        return sample

    def image_preprocess(self, img: Any) -> Any:
        raise NotImplementedError


def identity(x: Any) -> Any:
    return x
