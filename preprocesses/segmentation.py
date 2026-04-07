import gc
from typing import Optional

import cv2
import numpy as np
import torch
import torch.cuda
import tqdm
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

import models.grounded_sam.model
from data import resolve_model_path
from models.config import GroundedSAMConfig
from pipelines.scene import Scene
from preprocesses.config import SegmentationConfig


class Segmentator:
    @torch.inference_mode()
    def __call__(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError


class GroundedSAMSegmentator(Segmentator):
    def __init__(self, conf: GroundedSAMConfig, device: torch.device):
        object_detector = pipeline(
            model=str(resolve_model_path(conf.detector.pretrained_model)),
            task="zero-shot-object-detection",
            device=device,
        )
        segmentator = AutoModelForMaskGeneration.from_pretrained(
            str(resolve_model_path(conf.segmentator.pretrained_model))
        ).to(device)
        segmentation_processor = AutoProcessor.from_pretrained(
            str(resolve_model_path(conf.segmentator.pretrained_model))
        )

        self.conf = conf
        self.object_detector = object_detector
        self.segmentator = segmentator
        self.segmentation_processor = segmentation_processor

    @torch.inference_mode()
    def __call__(self, image: Image.Image) -> np.ndarray:
        _, detection_results = models.grounded_sam.model.grounded_segmentation(
            image,
            self.conf.labels,
            threshold=self.conf.threshold,
            polygon_refinement=True,
            object_detector=self.object_detector,
            segmentator=self.segmentator,
            segmentation_processor=self.segmentation_processor,
        )

        if self.conf.mode == "neg-and-pos":
            mask = np.ones((image.height, image.width), dtype=bool)
            for r in detection_results:
                if r.label in self.conf.negative_labels:
                    if r.mask is not None:
                        # r.mask > 0 means negative-class pixels
                        mask = mask & (r.mask == 0)

            for r in detection_results:
                if r.label in self.conf.positive_labels:
                    if r.mask is not None:
                        mask = mask | (r.mask > 0)
            mask = mask.astype(np.uint8)
        elif self.conf.mode == "pos-only":
            mask = np.zeros((image.height, image.width), dtype=bool)
            for r in detection_results:
                if r.label in self.conf.positive_labels:
                    if r.mask is not None:
                        mask = mask | (r.mask > 0)
            mask = mask.astype(np.uint8)
        else:
            raise ValueError(self.conf.mode)

        torch.cuda.empty_cache()
        return mask


def create_segmentator(
    conf: SegmentationConfig, device: Optional[torch.device] = None
) -> Segmentator:
    device = device or torch.device("cuda")
    if conf.type == "grounded_sam":
        assert conf.grounded_sam
        segmentator = GroundedSAMSegmentator(conf.grounded_sam, device=device)
    else:
        raise ValueError
    return segmentator


@torch.inference_mode()
def run_segmentation(
    scene: Scene,
    conf: SegmentationConfig,
    device: torch.device,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> Scene:
    is_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)

    segmentator = create_segmentator(conf, device=device)
    for i, path in enumerate(scene.image_paths):
        img = scene.get_image(path, use_original=True)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        H = image.height
        W = image.width
        # Resize
        image.thumbnail((conf.resize, conf.resize))

        try:
            mask = segmentator(image)

            _mask = torch.from_numpy(mask)[None][None]
            _mask = torch.nn.functional.interpolate(
                _mask, size=(H, W), mode="nearest"
            )
            mask = _mask.cpu().numpy()
            mask = mask[0][0]
        except Exception as e:
            print(f'Segmentation failed: {e}')
            mask = np.ones((H, W), dtype=np.uint8)

        if conf.dilation is not None:
            assert conf.dilation > 0
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=conf.dilation)

        scene.update_segmentation_mask_image(path, mask_image=mask)

        if progress_bar:
            progress_bar.set_postfix_str(
                f"Segmentation: ({i}/{len(scene.image_paths)})"
            )

    del segmentator
    segmentator = None

    gc.collect()
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(is_deterministic)
    return scene
