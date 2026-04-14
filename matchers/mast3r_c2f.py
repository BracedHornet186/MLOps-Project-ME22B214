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
from dust3r.utils.geometry import geotrf
from dust3r.utils.image import (
    ImgNorm,
    _resize_pil_image,
    exif_transpose,
    heif_support_enabled,
    load_images,
)
from mast3r.fast_nn import extract_correspondences_nonsym, fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.coarse_to_fine import crop_slice, select_pairs_of_crops
from PIL import Image

from data import FilePath, resolve_model_path
from matchers.base import DetectorFreeMatcher
from matchers.config import MASt3RC2FMatcherConfig
from models.mast3r.model import get_mast3r_model
from models.mast3r.visloc import (
    coarse_matching,
    do_crop,
    fine_matching,
    preprocess_view,
    resize_image_to_max,
)
from models.mast3r.visloc_utils import get_HW_resolution, rescale_points3d
from preprocesses.region import OverlapRegionCropper
from storage import MatchedKeypointStorage


def read_image(path: str) -> np.ndarray:
    return cv2.imread(str(path))


class MASt3RC2FMatcher(DetectorFreeMatcher):
    def __init__(
        self, conf: MASt3RC2FMatcherConfig, device: torch.device | None = None
    ):
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
            raise NotImplementedError

        # maxdim = max(self.model.patch_embed.img_size)
        maxdim = self.size
        patch_size = self.model.patch_embed.patch_size
        fast_nn_params = dict(device=self.device, dist="dot", block_size=2**13)

        view1 = preprocess_view(path1, 1, maxdim, patch_size, image_reader=image_reader)
        view2 = preprocess_view(path2, 2, maxdim, patch_size, image_reader=image_reader)

        query_rgb_tensor, query_to_orig_max, query_to_resize_max, (HQ, WQ) = (
            resize_image_to_max(self.conf.max_image_size, view1["rgb"])
        )
        # Pairs of crops have the same resolution
        query_resolution = get_HW_resolution(
            HQ, WQ, maxdim=maxdim, patchsize=patch_size
        )

        # print(maxdim, self.size, WQ, HQ)

        # Coarse matching
        if maxdim < max(WQ, HQ):
            # Use all points
            _, coarse_matches_im0, coarse_matches_im1, _ = coarse_matching(
                view1,
                view2,
                self.model,
                self.device,
                0,
                fast_nn_params,
                use_amp=self.conf.mast3r.use_amp,
                subsample=self.conf.subsample,
            )

            # valid_all = view2["valid"]
            # pts3d = view2["pts3d"]
            valid_all = None
            pts3d = None

            WM_full, HM_full = view2["rgb"].size
            (
                map_rgb_tensor,
                map_to_orig_max,
                map_to_resize_max,
                (HM, WM),
            ) = resize_image_to_max(self.conf.max_image_size, view2["rgb"])

            if WM_full != WM or HM_full != HM:
                # y_full, x_full = torch.where(valid_all)
                # pos2d_cv2 = (
                #     torch.stack([x_full, y_full], dim=-1)
                #     .cpu()
                #     .numpy()
                #     .astype(np.float64)
                # )
                # sparse_pts3d = pts3d[y_full, x_full].cpu().numpy()
                # _, _, pts3d_max, valid_max = rescale_points3d(
                #     pos2d_cv2, sparse_pts3d, map_to_resize_max, HM, WM
                # )
                # pts3d = torch.from_numpy(pts3d_max)
                # valid_all = torch.from_numpy(valid_max)
                pass

            coarse_matches_im0 = geotrf(
                query_to_resize_max, coarse_matches_im0, norm=True
            )
            coarse_matches_im1 = geotrf(
                map_to_resize_max, coarse_matches_im1, norm=True
            )

            crops1, crops2 = [], []
            crops_v1, crops_p1 = [], []
            to_orig1, to_orig2 = [], []
            map_resolution = get_HW_resolution(
                HM, WM, maxdim=maxdim, patchsize=self.model.patch_embed.patch_size
            )
            for crop_q, crop_b, pair_tag in select_pairs_of_crops(
                map_rgb_tensor,
                query_rgb_tensor,
                coarse_matches_im1,
                coarse_matches_im0,
                maxdim=maxdim,
                overlap=0.5,
                forced_resolution=[map_resolution, query_resolution],
            ):
                map_K = None
                query_K = None

                c1, v1, p1, trf1 = do_crop(
                    map_rgb_tensor, valid_all, pts3d, crop_q, map_K
                )
                c2, _, _, trf2 = do_crop(query_rgb_tensor, None, None, crop_b, query_K)
                crops1.append(c1)
                crops2.append(c2)
                crops_v1.append(v1)
                crops_p1.append(p1)
                to_orig1.append(trf1)
                to_orig2.append(trf2)

            if len(crops1) == 0 or len(crops2) == 0:
                valid_pts3d, matches_im_query, matches_im_map, matches_conf = (
                    [],
                    [],
                    [],
                    [],
                )
            else:
                crops1, crops2 = torch.stack(crops1), torch.stack(crops2)
                if len(crops1.shape) == 3:
                    crops1, crops2 = crops1[None], crops2[None]
                # crops_v1 = torch.stack(crops_v1)
                # crops_p1 = torch.stack(crops_p1)
                crops_v1 = None
                crops_p1 = None
                to_orig1, to_orig2 = (
                    torch.stack(to_orig1),
                    torch.stack(to_orig2),
                )
                map_crop_view = dict(
                    img=crops1.permute(0, 3, 1, 2),
                    instance=["1" for _ in range(crops1.shape[0])],
                    # valid=crops_v1,
                    # pts3d=crops_p1,
                    to_orig=to_orig1,
                )
                query_crop_view = dict(
                    img=crops2.permute(0, 3, 1, 2),
                    instance=["2" for _ in range(crops2.shape[0])],
                    to_orig=to_orig2,
                )

                # print(map_crop_view["img"].shape, query_crop_view["img"].shape)

                # Inference and Matching
                valid_pts3d, matches_im_query, matches_im_map, matches_conf = (
                    fine_matching(
                        query_crop_view,
                        map_crop_view,
                        self.model,
                        self.device,
                        self.conf.max_batch_size,
                        self.conf.pixel_tol,
                        fast_nn_params,
                        use_amp=self.conf.mast3r.use_amp,
                        subsample=self.conf.subsample,
                    )
                )
                matches_im_query = geotrf(
                    query_to_orig_max, matches_im_query, norm=True
                )
                matches_im_map = geotrf(map_to_orig_max, matches_im_map, norm=True)
        else:
            # use only valid 2d points
            valid_pts3d, matches_im_query, matches_im_map, matches_conf = (
                coarse_matching(
                    view1,
                    view2,
                    self.model,
                    self.device,
                    self.conf.pixel_tol,
                    fast_nn_params,
                    use_amp=self.conf.mast3r.use_amp,
                    subsample=self.conf.subsample,
                )
            )

        # apply conf
        if len(matches_conf) == 0:
            return

        # NOTE
        # matches_im_query: (N, 2)
        # matches_im_map: (N, 2)
        # matches_conf: (N,)
        # print(
        #     # valid_pts3d.shape,
        #     matches_im_query.shape,
        #     matches_im_map.shape,
        #     matches_conf.shape,
        # )

        assert isinstance(matches_conf, np.ndarray)
        assert isinstance(matches_im_query, np.ndarray)
        assert isinstance(matches_im_map, np.ndarray)
        # valid_pts3d = valid_pts3d[mask]
        mask = matches_conf >= self.conf.match_threshold
        mkpts1 = matches_im_query[mask]
        mkpts2 = matches_im_map[mask]
        scores = matches_conf[mask]

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

        if cropper:
            raise NotImplementedError

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
