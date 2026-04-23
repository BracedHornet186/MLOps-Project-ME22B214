from __future__ import annotations

import gc
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import tqdm
from PIL import Image
from vggt.models.vggt import VGGT
from vggt.utils.visual_track import visualize_tracks_on_images

from scripts.data import resolve_model_path
from extractor import LocalFeatureExtractor, extract_all
from features.factory import create_local_feature_handler
from matchers.base import PointTrackingMatcher
from matchers.config import VGGTMatcherConfig
from models.vggt.custom_model import CustomVGGT
from pipelines.scene import Scene
from preprocesses.region import OverlapRegionCropper
from storage import (
    InMemoryKeypointStorage,
    InMemoryLocalFeatureStorage,
    KeypointStorage,
    LocalFeatureStorage,
    MatchedKeypointStorage,
    MatchingStorage,
    concat_keypoints,
)


class VGGTMatcher(PointTrackingMatcher):
    def __init__(
        self,
        conf: VGGTMatcherConfig,
        extractors: list[LocalFeatureExtractor],
        device: torch.device,
    ):
        self.conf = conf
        self._extractors = extractors
        if self.conf.load_fixed_weight:
            model = VGGT()
            model.load_state_dict(
                torch.load(resolve_model_path(conf.model.pretrained_model))
            )
            self.model = model.eval().to(device)
        else:
            self.model = (
                VGGT.from_pretrained(resolve_model_path(conf.model.pretrained_model))
                .eval()
                .to(device)
            )

        self.device = device
        self.slim()

    def slim(self):
        torch.cuda.synchronize()
        del self.model.camera_head
        del self.model.point_head
        del self.model.depth_head
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def extractors(self) -> list[LocalFeatureExtractor]:
        return self._extractors

    @torch.inference_mode()
    def __call__(
        self,
        path1: str | Path,
        path2: str | Path,
        keypoint_storage: KeypointStorage,
        matching_storage: MatchingStorage,  # TODO: support
        matched_keypoint_storage: MatchedKeypointStorage,
        cropper: OverlapRegionCropper | None = None,
        orientation1: int | None = None,
        orientation2: int | None = None,
        image_reader: Callable[..., Any] | None = None,
    ):
        track_list, vis_score, conf_score = self.predict_vggt_track(
            path1,
            path2,
            keypoint_storage,
            image_reader=image_reader,
        )
        tracks = track_list[-1]  # Shape(S, N, 2)
        mask = self.create_vggt_track_mask(vis_score, conf_score)

        if len(tracks.shape) == 4:
            tracks = tracks.squeeze(0)
            mask = mask.squeeze(0)

        assert (
            len(tracks) == 2
        )  # Assume that VGGT computes tracks between image1 and image2
        assert len(mask) == 2

        track1 = tracks[0]
        track2 = tracks[1]
        mask2 = mask[1]
        valid2 = torch.where(mask2)[0]
        mkpts1 = track1[valid2].cpu().numpy()
        mkpts2 = track2[valid2].cpu().numpy()

        if len(mkpts1) == 0:
            mkpts1 = np.empty((0, 2), dtype=np.float32)
            mkpts2 = np.empty((0, 2), dtype=np.float32)
        else:
            pass

        if self.conf.min_matches is None or len(mkpts1) >= self.conf.min_matches:
            scores = np.ones((len(mkpts1),))
            matched_keypoint_storage.add(path1, path2, mkpts1, mkpts2, scores=scores)

    @torch.inference_mode()
    def predict_vggt_track(
        self,
        path1: str | Path,
        path2: str | Path,
        keypoint_storage: KeypointStorage,
        coordinate_type: Literal["origin", "vggt"] = "origin",
        image_reader: Callable[..., Any] | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        kpts1 = keypoint_storage.get(path1)
        if kpts1 is None or len(kpts1) == 0:
            return (
                [],
                torch.empty((0, 2, 0), dtype=torch.float32),
                torch.empty((0, 2, 0), dtype=torch.float32),
            )

        images, pads, resize_shapes, origin_shapes = load_and_preprocess_images_imc25(
            [path1, path2],
            mode="pad",
            image_reader=image_reader,
            target_size=self.conf.target_size,
        )
        images = images.to(self.device, non_blocking=True)

        with torch.autocast(self.device.type):
            x = images[None]  # Add batch dim
            aggregated_tokens_list, ps_idx = self.model.aggregator(x)

        query_points = torch.from_numpy(kpts1).to(self.device)
        origin_h1, origin_w1 = origin_shapes[0]
        resize_h1, resize_w1 = resize_shapes[0]
        pad_left1, pad_top1 = pads[0]

        query_points[:, 0] = (query_points[:, 0] / origin_w1) * resize_w1 + pad_left1
        query_points[:, 1] = (query_points[:, 1] / origin_h1) * resize_h1 + pad_top1

        with torch.autocast(self.device.type):
            track_list, vis_score, conf_score = self.model.track_head(
                aggregated_tokens_list,
                x,
                ps_idx,
                query_points=query_points[None],
                iters=self.conf.track_iters,
            )

        if coordinate_type == "vggt":
            # NOTE
            # Track coordinates (x,y) are based on an image coordinates after resizing and padding.
            return track_list, vis_score, conf_score

        origin_h2, origin_w2 = origin_shapes[1]
        resize_h2, resize_w2 = resize_shapes[1]
        pad_left2, pad_top2 = pads[1]

        for i in range(len(track_list)):
            track = track_list[i]
            # NOTE
            # track: Shape(B=1, S=2, #kpts, 2)
            assert track.shape[1] == 2
            # For image1
            track[..., 0, :, 0] = (
                (track[..., 0, :, 0] - pad_left1) / resize_w1
            ) * origin_w1
            track[..., 0, :, 1] = (
                (track[..., 0, :, 1] - pad_top1) / resize_h1
            ) * origin_h1
            # For image2
            track[..., 1, :, 0] = (
                (track[..., 1, :, 0] - pad_left2) / resize_w2
            ) * origin_w2
            track[..., 1, :, 1] = (
                (track[..., 1, :, 1] - pad_top2) / resize_h2
            ) * origin_h2
            track_list[i] = track

        return track_list, vis_score, conf_score

    @torch.inference_mode()
    def visualize_vggt_track(
        self,
        path1: str | Path,
        path2: str | Path,
        keypoint_storage: KeypointStorage,
    ):
        images, pads, resize_shapes, origin_shapes = load_and_preprocess_images_imc25(
            [path1, path2],
            mode="pad",
            target_size=self.conf.target_size,
        )
        track_list, vis_score, conf_score = self.predict_vggt_track(
            path1, path2, keypoint_storage
        )
        track = track_list[-1]
        mask = self.create_vggt_track_mask(vis_score, conf_score)
        visualize_tracks_on_images(images, track, mask, out_dir="track_visuals")

    def create_vggt_track_mask(
        self, vis_score: torch.Tensor, conf_score: torch.Tensor
    ) -> torch.Tensor:
        if self.conf.filtering_method == "conf":
            return conf_score >= self.conf.conf_score_threshold
        elif self.conf.filtering_method == "vis":
            return vis_score >= self.conf.vis_score_threshold
        elif self.conf.filtering_method == "conf&vis":
            return (conf_score >= self.conf.conf_score_threshold) & (
                vis_score >= self.conf.vis_score_threshold
            )
        elif self.conf.filtering_method == "conf|vis":
            return (conf_score >= self.conf.conf_score_threshold) | (
                vis_score >= self.conf.vis_score_threshold
            )
        else:
            raise ValueError(self.conf.filtering_method)


class CustomVGGTMatcher(VGGTMatcher):
    def __init__(
        self,
        conf: VGGTMatcherConfig,
        extractors: list[LocalFeatureExtractor],
        device: torch.device,
    ):
        self.conf = conf
        self._extractors = extractors
        self.model = (
            CustomVGGT.from_pretrained(resolve_model_path(conf.model.pretrained_model))
            .eval()
            .to(device)
        )
        print("Use CustomVGGT")
        self.device = device
        self.slim()
        self.patch_token_caches: dict[str, torch.Tensor] = {}
        self.input_info_caches: dict[
            str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        ] = {}

    @torch.inference_mode()
    def prepare(
        self,
        image_paths: Sequence[str | Path],
        image_reader: Callable[..., Any] | None = None,
        progress_bar: tqdm.tqdm | None = None,
    ) -> None:
        assert isinstance(self.model, CustomVGGT)
        for i, path in enumerate(image_paths):
            images, pads, resize_shapes, origin_shapes = (
                load_and_preprocess_images_imc25(
                    [path],
                    mode="pad",
                    image_reader=image_reader,
                    target_size=self.conf.target_size,
                )
            )
            assert (
                len(pads) == 1 and len(resize_shapes) == 1 and len(origin_shapes) == 1
            )
            images = images.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type):
                patch_tokens = self.model.extract_patch_tokens(images)
            self.patch_token_caches[str(path)] = patch_tokens.detach()  # Shape(1, P, C)
            self.input_info_caches[str(path)] = (
                pads[0],
                resize_shapes[0],
                origin_shapes[0],
            )
            if progress_bar:
                progress_bar.set_postfix_str(
                    f"CustomVGGTMatcher.prepare ({i + 1}/{len(image_paths)})"
                )

    @torch.inference_mode()
    def predict_vggt_track(
        self,
        path1: str | Path,
        path2: str | Path,
        keypoint_storage: KeypointStorage,
        coordinate_type: Literal["origin", "vggt"] = "origin",
        image_reader: Callable[..., Any] | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        kpts1 = keypoint_storage.get(path1)
        if kpts1 is None or len(kpts1) == 0:
            return (
                [],
                torch.empty((0, 2, 0), dtype=torch.float32),
                torch.empty((0, 2, 0), dtype=torch.float32),
            )

        start_time_a = time.time()
        pad1, resize_shape1, origin_shape1 = self.input_info_caches[str(path1)]
        pad2, resize_shape2, origin_shape2 = self.input_info_caches[str(path2)]
        patch_tokens1 = self.patch_token_caches[str(path1)]
        patch_tokens2 = self.patch_token_caches[str(path2)]
        patch_tokens = torch.vstack([patch_tokens1, patch_tokens2])
        # patch_tokens = patch_tokens.to(self.device, non_blocking=True)

        pads = [pad1, pad2]
        resize_shapes = [resize_shape1, resize_shape2]
        origin_shapes = [origin_shape1, origin_shape2]

        elapsed_time_a = time.time() - start_time_a
        start_time_b = time.time()

        with torch.autocast(self.device.type):
            aggregated_tokens_list, ps_idx = self.model.aggregator(patch_tokens)

        torch.cuda.synchronize()
        elapsed_time_b = time.time() - start_time_b

        query_points = torch.from_numpy(kpts1).to(self.device)
        origin_h1, origin_w1 = origin_shapes[0]
        resize_h1, resize_w1 = resize_shapes[0]
        pad_left1, pad_top1 = pads[0]

        query_points[:, 0] = (query_points[:, 0] / origin_w1) * resize_w1 + pad_left1
        query_points[:, 1] = (query_points[:, 1] / origin_h1) * resize_h1 + pad_top1

        start_time_c = time.time()

        with torch.autocast(self.device.type):
            track_list, vis_score, conf_score = self.model.track_head(
                aggregated_tokens_list,
                ps_idx,
                query_points=query_points[None],
                iters=self.conf.track_iters,
            )

        torch.cuda.synchronize()
        elapsed_time_c = time.time() - start_time_c
        print("time", elapsed_time_a, elapsed_time_b, elapsed_time_c)

        if coordinate_type == "vggt":
            # NOTE
            # Track coordinates (x,y) are based on an image coordinates after resizing and padding.
            return track_list, vis_score, conf_score

        origin_h2, origin_w2 = origin_shapes[1]
        resize_h2, resize_w2 = resize_shapes[1]
        pad_left2, pad_top2 = pads[1]

        for i in range(len(track_list)):
            track = track_list[i]
            # NOTE
            # track: Shape(B=1, S=2, #kpts, 2)
            assert track.shape[1] == 2
            # For image1
            track[..., 0, :, 0] = (
                (track[..., 0, :, 0] - pad_left1) / resize_w1
            ) * origin_w1
            track[..., 0, :, 1] = (
                (track[..., 0, :, 1] - pad_top1) / resize_h1
            ) * origin_h1
            # For image2
            track[..., 1, :, 0] = (
                (track[..., 1, :, 0] - pad_left2) / resize_w2
            ) * origin_w2
            track[..., 1, :, 1] = (
                (track[..., 1, :, 1] - pad_top2) / resize_h2
            ) * origin_h2
            track_list[i] = track

        return track_list, vis_score, conf_score


def load_and_preprocess_images_imc25(
    image_path_list: list[str | Path],
    mode: Literal["crop", "pad"] = "crop",
    image_reader: Callable[..., Any] | None = None,
    target_size: int = 518,
) -> tuple[
    torch.Tensor, list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]
]:
    """
    Adapted from vggt.utils.load_fn.load_and_preprocess_images()
    ---------

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    pads: list[tuple[int, int]] = []  # [(pad_left, pad_top), ...]
    origin_shapes: list[tuple[int, int]] = []  # [(H, W), ...]
    resize_shapes: list[tuple[int, int]] = []  # [(H, W), ...]

    images = []
    shapes = set()
    to_tensor = T.ToTensor()
    # target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        if image_reader is None:
            img = Image.open(image_path)
        else:
            cached_img = image_reader(str(image_path))
            if isinstance(cached_img, np.ndarray):
                cached_img = cv2.cvtColor(cached_img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cached_img)
            else:
                raise NotImplementedError

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = (
                    round(height * (new_width / width) / 14) * 14
                )  # Make divisible by 14
            else:
                new_height = target_size
                new_width = (
                    round(width * (new_height / height) / 14) * 14
                )  # Make divisible by 14
        else:  # mode == "crop"
            raise NotImplementedError

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            raise NotImplementedError

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            else:
                pad_left = 0
                pad_top = 0

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
        pads.append((pad_left, pad_top))
        origin_shapes.append((height, width))
        resize_shapes.append((new_height, new_width))

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for i, img in enumerate(images):
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            else:
                pad_left = 0
                pad_top = 0
            padded_images.append(img)
            pads[i] = (pads[i][0] + pad_left, pads[i][1] + pad_top)

        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images, pads, resize_shapes, origin_shapes
