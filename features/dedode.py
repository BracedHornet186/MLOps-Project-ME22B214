from typing import Callable, Optional

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn
from kornia.core import Module, Tensor
from kornia.enhance.normalize import Normalize
from kornia.feature import DeDoDe
from kornia.feature.dedode.dedode_models import get_detector, get_descriptor
from kornia.feature.dedode.detector import DeDoDeDetector
from kornia.feature.dedode.decoder import Decoder, ConvRefiner
from kornia.feature.dedode.descriptor import DeDoDeDescriptor
from kornia.feature.dedode.encoder import VGG_DINOv2
from kornia.utils.image import image_to_tensor
from pydantic import BaseModel

from data import FilePath, resolve_model_path
from features.base import (
    LocalFeatureHandler,
    LocalFeatureOutputs,
    postprocess,
    read_image,
)
from features.config import DeDoDeConfig
from preprocess import resize_image_tensor
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class DeDoDeHandler(LocalFeatureHandler):
    def __init__(self, conf: DeDoDeConfig, device: Optional[torch.device] = None):
        self.conf = conf
        self.device = device
        model = from_pretrained(conf)
        self.model = model.eval().to(device)

    def __call__(
        self,
        path: FilePath,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image,
    ) -> LocalFeatureOutputs:
        if rotation:
            raise NotImplementedError
        img = image_reader(str(path))

        if cropper:
            cropper.set_original_image(img)
            img = cropper.crop_ndarray_image(img)

        x = self.to_torch_image(img)
        x = self.preprocess(x, resize=resize)
        outputs = self.extract(x)
        outputs = postprocess(outputs, img, x, cropper=cropper)
        return outputs

    def to_torch_image(self, img: np.ndarray) -> torch.Tensor:
        x = image_to_tensor(img, False).float() / 255.0
        x = kornia.color.bgr_to_rgb(x)
        return x

    def preprocess(
        self, x: torch.Tensor, resize: Optional[ResizeConfig] = None
    ) -> torch.Tensor:
        if resize is not None:
            x = resize_image_tensor(x, resize)
        return x

    def extract(self, x: torch.Tensor, *args, **kwargs) -> LocalFeatureOutputs:
        x = x.to(self.device)

        kpts, scores, descs = self.model(
            x,
            n=self.conf.num_features,
            apply_imagenet_normalization=True,
            pad_if_not_divisible=True,
        )
        kpts = kpts[0]  # Shape(1, N, 2) -> Shape(N, 2)
        scores = scores[0]  # Shape(1, N) -> Shape(N,)
        descs = descs[0]  # Shape(1, N, 256) -> Shape(N, 256)

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]
        return lafs, scores, descs


class _DeDoDe(DeDoDe):
    def __init__(
        self,
        dinov2_weight_path: Optional[str] = None,
        detector_model: str = "L",
        descriptor_model: str = "G",
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        Module.__init__(self)
        self.detector: DeDoDeDetector = get_detector(detector_model, amp_dtype)
        if descriptor_model == "G":
            assert dinov2_weight_path
            dinov2_weight_path = str(resolve_model_path(dinov2_weight_path))
            dinov2_weights = torch.load(dinov2_weight_path, map_location="cpu")
            self.descriptor: DeDoDeDescriptor = dedode_descriptor_G(
                dinov2_weights, amp_dtype=amp_dtype
            )
        elif descriptor_model == "B":
            self.descriptor: DeDoDeDescriptor = get_descriptor(
                descriptor_model, amp_dtype
            )
        else:
            raise ValueError
        self.normalizer = Normalize(
            torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])
        )


def from_pretrained(
    conf: DeDoDeConfig,
    amp_dtype: torch.dtype = torch.float16,
) -> DeDoDe:
    model = _DeDoDe(
        dinov2_weight_path=conf.dinov2_weight_path,
        detector_model=conf.detector_model_name[0],  # type: ignore[arg-type]
        descriptor_model=conf.descriptor_model_name[0],  # type: ignore[arg-type]
        amp_dtype=amp_dtype,
    )
    model.detector.load_state_dict(
        torch.load(resolve_model_path(conf.detector_weight_path), map_location="cpu")
    )
    print(f"[DeDoDe] detector: {resolve_model_path(conf.detector_weight_path)}")
    model.descriptor.load_state_dict(
        torch.load(resolve_model_path(conf.descriptor_weight_path), map_location="cpu")
    )
    print(f"[DeDoDe] descriptor: {resolve_model_path(conf.descriptor_weight_path)}")
    model.eval()
    return model


def dedode_descriptor_G(
    dinov2_weights: str, amp_dtype: torch.dtype = torch.float16
) -> DeDoDeDescriptor:
    NUM_PROTOTYPES = 256  # == descriptor size
    residual = True
    hidden_blocks = 5
    amp = True
    conv_refiner = nn.ModuleDict(
        {
            "14": ConvRefiner(
                1024,
                768,
                512 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "8": ConvRefiner(
                512 + 512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                64,
                32 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
        }
    )
    vgg_kwargs = {"amp": amp, "amp_dtype": amp_dtype}
    dinov2_kwargs = {
        "amp": amp,
        "amp_dtype": amp_dtype,
        "dinov2_weights": dinov2_weights,
    }
    encoder = VGG_DINOv2(vgg_kwargs=vgg_kwargs, dinov2_kwargs=dinov2_kwargs)
    decoder = Decoder(conv_refiner, num_prototypes=NUM_PROTOTYPES)
    model = DeDoDeDescriptor(encoder=encoder, decoder=decoder)
    return model
