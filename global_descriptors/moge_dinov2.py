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
from moge.utils.vis import colorize_depth
from torch.utils.data.dataset import Dataset
from transformers import AutoImageProcessor, AutoModel

from data import FilePath, resolve_model_path
from global_descriptors.base import CustomDescriptorExtractor
from global_descriptors.dinov2 import DINOv2GlobalDescriptorExtractor
from models.config import HuggingFaceModelConfig, MoGeModelConfig
from models.moge.feature import MoGeFeatureModel
from pipelines.scene import Scene


class MoGeDepthDINOv2GlobalDescriptorExtractor(CustomDescriptorExtractor):
    def __init__(
        self,
        conf: MoGeModelConfig,
        dinov2_extractor: DINOv2GlobalDescriptorExtractor,
        device: torch.device | None = None,
    ):
        self.conf = conf
        self.device = device or torch.device("cuda")
        self.model = (
            MoGeModel.from_pretrained("Ruicheng/moge-vitl").eval().to(self.device)
        )
        self.dinov2_extractor = dinov2_extractor
        self.dinov2_processor = AutoImageProcessor.from_pretrained(
            resolve_model_path(self.dinov2_extractor.conf.pretrained_model)
        )
        self.descriptor = cv2.HOGDescriptor()

    @torch.inference_mode()
    def __call__(self, x: Any) -> torch.Tensor:
        assert len(x) == 1, "Required: batch_size=1"
        x = x.to(self.device, non_blocking=True)
        with torch.autocast(self.device.type):
            output = self.model.infer(
                x,
                fov_x=None,  # type: ignore
                resolution_level=9,
                num_tokens=None,  # type: ignore
                use_fp16=True,
            )
        points, depth, mask, intrinsics = (
            output["points"].cpu().numpy(),
            output["depth"].cpu().numpy(),
            output["mask"].cpu().numpy(),
            output["intrinsics"].cpu().numpy(),
        )
        depth_image = colorize_depth(depth)  # Shape(1, H, W, 3)
        depth_image_tensor = torch.from_numpy(depth_image)
        depth_image_tensor = depth_image_tensor.permute(0, 3, 1, 2)
        depth_image_tensor = self.dinov2_processor(
            images=depth_image_tensor, return_tensors="pt", do_rescale=False
        )

        feat = self.dinov2_extractor(depth_image_tensor["pixel_values"])
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
        cls, scene: Scene, extractor: MoGeDepthDINOv2GlobalDescriptorExtractor
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
