from __future__ import annotations

import os
from collections.abc import Callable
from typing import Optional

import cv2
import numpy as np
import torch
import torch.cuda.amp
import tqdm
from dust3r.inference import (
    check_if_same_size,
    collate_with_cat,
    loss_of_one_batch,
    to_cpu,
)
from dust3r.utils.image import (
    ImgNorm,
    _resize_pil_image,
    exif_transpose,
    heif_support_enabled,
    load_images,
)
from mast3r.fast_nn import extract_correspondences_nonsym, fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R
from PIL import Image

from scripts.data import FilePath, resolve_model_path
from matchers.base import DetectorFreeMatcher
from matchers.config import MASt3RMatcherConfig
from models.mast3r.model import get_mast3r_model
from preprocesses.region import OverlapRegionCropper
from scripts.storage import MatchedKeypointStorage


def read_image(path: str) -> np.ndarray:
    return cv2.imread(str(path))


class MASt3RMatcher(DetectorFreeMatcher):
    def __init__(self, conf: MASt3RMatcherConfig, device: torch.device | None = None):
        assert device is not None
        self.conf = conf
        self.device = device
        self.size = conf.size

        print(f"MASt3R: size={self.size}")

        self.model = get_mast3r_model(
            resolve_model_path(conf.mast3r.weight_path), self.device
        )

    @torch.inference_mode()
    def __call__(
        self,
        path1: FilePath,
        path2: FilePath,
        matched_keypoint_storage: MatchedKeypointStorage,
        cropper: Optional[OverlapRegionCropper] = None,
        orientation1: Optional[int] = None,
        orientation2: Optional[int] = None,
        image_reader: Callable = read_image,
    ):
        orig_img1 = image_reader(path1)
        orig_img2 = image_reader(path2)

        if cropper:
            orig_img1 = cv2.cvtColor(orig_img1, cv2.COLOR_BGR2RGB)
            orig_img2 = cv2.cvtColor(orig_img2, cv2.COLOR_BGR2RGB)
            cropper.set_original_image(orig_img1, orig_img2)

        # NOTE: Read images again without caches
        paired_images = load_images_fixed(
            [str(path1), str(path2)],
            size=self.size,
            verbose=False,
            image_reader=image_reader,
            cropper=cropper,
        )
        crop_offset1 = paired_images[0].pop("crop_offset")
        crop_offset2 = paired_images[1].pop("crop_offset")
        size_before_crop1 = paired_images[0].pop("size_before_crop")
        size_before_crop2 = paired_images[1].pop("size_before_crop")
        _ = paired_images[0].pop("crop_center")
        _ = paired_images[1].pop("crop_center")
        _ = paired_images[0].pop("half_wh")
        _ = paired_images[1].pop("half_wh")

        output: dict[str, dict[str, torch.Tensor]] = inference_fixed(
            [tuple(paired_images)],
            self.model,
            self.device,
            batch_size=1,
            use_amp=self.conf.mast3r.use_amp,
            verbose=False,
        )  # type: ignore

        view1, pred1 = output["view1"], output["pred1"]
        view2, pred2 = output["view2"], output["pred2"]

        # NOTE
        # `pred1` has (`pts3d`, `conf`, `desc`, `desc_conf`) keys

        desc1, desc2 = (
            pred1["desc"].squeeze(0).detach(),
            pred2["desc"].squeeze(0).detach(),
        )

        conf1, conf2 = (
            pred1["desc_conf"].squeeze(0).detach(),
            pred2["desc_conf"].squeeze(0).detach(),
        )

        corres: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (
            extract_correspondences_nonsym(
                desc1,
                desc2,
                conf1,
                conf2,
                device=self.device,
                subsample=self.conf.subsample,
                pixel_tol=self.conf.pixel_tol,
            )
        )  # type: ignore
        score = corres[2]
        mask = score >= self.conf.match_threshold
        mkpts1 = corres[0][mask].cpu().numpy()
        mkpts2 = corres[1][mask].cpu().numpy()
        scores = score[mask].cpu().numpy()

        order = np.argsort(-scores)
        mkpts1 = mkpts1[order]
        mkpts2 = mkpts2[order]
        scores = scores[order]

        if self.conf.match_topk:
            mkpts1 = mkpts1[: self.conf.match_topk]
            mkpts2 = mkpts2[: self.conf.match_topk]
            scores = scores[: self.conf.match_topk]

        if len(mkpts1) == 0:
            mkpts1 = np.empty((0, 2), dtype=np.float32)
            mkpts2 = np.empty((0, 2), dtype=np.float32)
        else:
            mkpts1 = postprocess(
                mkpts1,
                orig_img1,
                paired_images[0]["true_shape"],
                crop_offset1,
                size_before_crop1,
            )
            mkpts2 = postprocess(
                mkpts2,
                orig_img2,
                paired_images[1]["true_shape"],
                crop_offset2,
                size_before_crop2,
            )

        if cropper:
            mkpts1, mkpts2 = cropper.convert_cropped_to_original_coordinates(
                mkpts1, mkpts2
            )

        if self.conf.min_matches is None or len(mkpts1) >= self.conf.min_matches:
            # scores = np.ones((len(mkpts1),))
            matched_keypoint_storage.add(path1, path2, mkpts1, mkpts2, scores=scores)


def postprocess(
    kpts: np.ndarray,
    orig_img: np.ndarray,
    resized_shape: np.ndarray,  # (H, W)
    crop_offset: tuple[int, int],  # (W, H)
    size_before_crop: tuple[int, int],  # (W, H)
) -> np.ndarray:
    # print(orig_img.shape, resized_shape)
    orig_H, orig_W = orig_img.shape[:2]
    # view_H, view_W = resized_shape[0]
    # scale_H = orig_H / view_H
    # scale_W = orig_W / view_W
    before_crop_resized_W, before_crop_resized_H = size_before_crop
    scale_H = orig_H / before_crop_resized_H
    scale_W = orig_W / before_crop_resized_W
    kpts = kpts.astype(np.float32)
    kpts[:, 0] += crop_offset[0]
    kpts[:, 1] += crop_offset[1]
    kpts[:, 0] *= scale_W
    kpts[:, 1] *= scale_H
    return kpts.astype(np.float32)  # NOTE: MASt3R returns keypoints as np.int64


@torch.no_grad()
def inference_fixed(
    pairs,
    model,
    device,
    batch_size: int = 8,
    use_amp: bool = False,
    verbose: bool = True,
):
    """dust3r.inference"""
    if verbose:
        print(f">> Inference with model on {len(pairs)} image pairs")
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch(
            collate_with_cat(pairs[i : i + batch_size]),
            model,
            None,
            device,
            use_amp=use_amp,
        )
        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    return result


def load_images_fixed(
    folder_or_list,
    size,
    square_ok=False,
    verbose=True,
    image_reader: Callable | None = None,
    cropper: Optional[OverlapRegionCropper] = None,
):
    """from dust3r.utils.image import load_images"""
    if isinstance(folder_or_list, str):
        if verbose:
            print(f">> Loading images from {folder_or_list}")
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f">> Loading a list of {len(folder_or_list)} images")
        root, folder_content = "", folder_or_list

    else:
        raise ValueError(f"bad {folder_or_list=} ({type(folder_or_list)})")

    supported_images_extensions = [".jpg", ".jpeg", ".png"]
    if heif_support_enabled:
        supported_images_extensions += [".heic", ".heif"]
    supported_images_extensions = tuple(supported_images_extensions)

    pre_load_imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue

        if image_reader:
            cvimg = image_reader(str(path))
            cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cvimg).convert("RGB")
        else:
            img = exif_transpose(Image.open(os.path.join(root, path))).convert("RGB")
        pre_load_imgs.append(img)

    if cropper:
        _img1 = pre_load_imgs[0]
        _img2 = pre_load_imgs[1]
        if isinstance(_img1, Image.Image):
            img1 = np.array(_img1)
        else:
            img1 = _img1
        if isinstance(_img2, Image.Image):
            img2 = np.array(_img2)
        else:
            img2 = _img2
        img1, img2 = cropper.crop_ndarray_image(img1, img2)
        pre_load_imgs = [
            Image.fromarray(img1).convert("RGB"),
            Image.fromarray(img2).convert("RGB"),
        ]

    imgs = []
    for img in pre_load_imgs:
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
            crop_offset = (cx - half, cy - half)
            halfw = half
            halfh = half
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not (square_ok) and W == H:
                halfh = 3 * halfw / 4
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
            crop_offset = (cx - halfw, cy - halfh)
        size_before_crop = (W, H)
        crop_center = (cx, cy)
        half_wh = (halfw, halfh)

        W2, H2 = img.size
        if verbose:
            print(
                f" - adding {path} with resolution {W1}x{H1} --> {W}x{H} --> {W2}x{H2}"
            )
            print(crop_offset)
        imgs.append(
            dict(
                img=ImgNorm(img)[None],  # type: ignore
                true_shape=np.int32([img.size[::-1]]),  # type: ignore
                idx=len(imgs),
                instance=str(len(imgs)),
                size_before_crop=size_before_crop,
                crop_offset=crop_offset,
                crop_center=crop_center,
                half_wh=half_wh,
            )
        )

    assert imgs, "no images foud at " + root
    if verbose:
        print(f" (Found {len(imgs)} images)")
    return imgs
