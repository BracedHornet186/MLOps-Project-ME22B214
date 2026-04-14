from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from moge.model.v1 import MoGeModel


class MoGeFeatureModel:
    def __init__(
        self,
        model: MoGeModel,
        device: torch.device,
        pooling_type: Literal["sum"] = "sum",
    ):
        self.model = model.eval().to(device)
        self.device = device
        self.pooling_type = pooling_type

    @torch.inference_mode()
    def extract_feature(
        self, path: str | Path, resolution_level: int = 9, num_tokens: int | None = None
    ):
        input_image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        input_image = torch.tensor(
            input_image / 255, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1)

        image = input_image.unsqueeze(0)
        return self(image, resolution_level=resolution_level, num_tokens=num_tokens)

    @torch.inference_mode()
    def __call__(
        self,
        image: torch.Tensor,
        resolution_level: int = 9,
        num_tokens: int | None = None,
    ) -> torch.Tensor:
        original_height, original_width = image.shape[-2:]
        area = original_height * original_width
        aspect_ratio = original_width / original_height

        if num_tokens is None:
            min_tokens, max_tokens = self.model.num_tokens_range
            num_tokens = int(
                min_tokens + (resolution_level / 9) * (max_tokens - min_tokens)
            )

        with torch.autocast(device_type=image.device.type, dtype=torch.float16):
            original_height, original_width = image.shape[-2:]

            # Resize to expected resolution defined by num_tokens
            resize_factor = (
                (num_tokens * 14**2) / (original_height * original_width)
            ) ** 0.5
            resized_width, resized_height = (
                int(original_width * resize_factor),
                int(original_height * resize_factor),
            )
            image = F.interpolate(
                image,
                (resized_height, resized_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

            # Apply image transformation for DINOv2
            image = (image - self.model.image_mean) / self.model.image_std
            image_14 = F.interpolate(
                image,
                (resized_height // 14 * 14, resized_width // 14 * 14),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # Get intermediate layers from the backbone
            hidden_states = self.model.backbone.get_intermediate_layers(
                image_14, self.model.intermediate_layers, return_class_token=True
            )

            img_h, img_w = image.shape[-2:]
            patch_h, patch_w = img_h // 14, img_w // 14

            # Process the hidden states
            x = torch.stack(
                [
                    proj(
                        feat.permute(0, 2, 1)
                        .unflatten(2, (patch_h, patch_w))
                        .contiguous()
                    )
                    for proj, (feat, clstoken) in zip(
                        self.model.head.projects, hidden_states
                    )
                ],
                dim=1,
            ).sum(dim=1)  # Shape(1, 512, 67, 37)

        if self.pooling_type == "sum":
            x = F.normalize(x.flatten(-2).permute(0, 2, 1).sum(dim=1))
        else:
            raise ValueError(self.pooling_type)
        return x
