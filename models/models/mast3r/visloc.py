from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import tqdm
from dust3r.inference import (
    check_if_same_size,
    collate_with_cat,
    loss_of_one_batch,
    to_cpu,
)
from dust3r.utils.geometry import geotrf, xy_grid
from dust3r.utils.image import ImgNorm
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.utils.coarse_to_fine import crop_slice
from mast3r.utils.collate import cat_collate, cat_collate_fn_map
from PIL import Image

from models.mast3r.model import AsymmetricMASt3R
from models.mast3r.visloc_utils import get_resize_function


def preprocess_view(
    path: str | Path,
    idx: int,
    maxdim: int,
    patch_size: int,
    image_reader: Callable = cv2.imread,
) -> dict:
    cv2img = image_reader(path)
    cv2img = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)

    rgb_image = Image.fromarray(cv2img).convert("RGB")
    W, H = rgb_image.size
    resize_func, to_resize, to_orig = get_resize_function(maxdim, patch_size, H, W)
    rgb_tensor = resize_func(ImgNorm(rgb_image))

    view = {
        "rgb": rgb_image,
        "rgb_rescaled": rgb_tensor,
        "to_orig": to_orig,
        "idx": idx,
        "image_name": Path(path).name,
    }
    return view


def resize_image_to_max(
    max_image_size: int, rgb: Image.Image
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, tuple[int, int]]:
    W, H = rgb.size
    if max_image_size and max(W, H) > max_image_size:
        islandscape = W >= H
        if islandscape:
            WMax = max_image_size
            HMax = int(H * (WMax / W))
        else:
            HMax = max_image_size
            WMax = int(W * (HMax / H))
        resize_op = T.Compose([ImgNorm, T.Resize(size=[HMax, WMax])])
        rgb_tensor: torch.Tensor = resize_op(rgb).permute(1, 2, 0)  # type: ignore
        to_orig_max = np.array([[W / WMax, 0, 0], [0, H / HMax, 0], [0, 0, 1]])
        to_resize_max = np.array([[WMax / W, 0, 0], [0, HMax / H, 0], [0, 0, 1]])

        # Generate new camera parameters
        # new_K = opencv_to_colmap_intrinsics(K)
        # new_K[0, :] *= WMax / W
        # new_K[1, :] *= HMax / H
        # new_K = colmap_to_opencv_intrinsics(new_K)
    else:
        rgb_tensor: torch.Tensor = ImgNorm(rgb).permute(1, 2, 0)  # type: ignore
        to_orig_max = np.eye(3)
        to_resize_max = np.eye(3)
        HMax, WMax = H, W
        # new_K = K
    return rgb_tensor, to_orig_max, to_resize_max, (HMax, WMax)


def do_crop(
    img: torch.Tensor,
    mask: torch.Tensor | None,
    pts3d: torch.Tensor | None,
    crop: np.ndarray,
    intrinsics=None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    out_cropped_img = img.clone()
    if mask is not None:
        out_cropped_mask = mask.clone()
    else:
        out_cropped_mask = None
    if pts3d is not None:
        out_cropped_pts3d = pts3d.clone()
    else:
        out_cropped_pts3d = None
    to_orig = torch.eye(3, device=img.device)

    # If intrinsics available, crop and apply rectifying homography. Otherwise, just crop
    if intrinsics is not None:
        raise NotImplementedError
    else:
        out_cropped_img = img[crop_slice(crop)]
        if out_cropped_mask is not None:
            out_cropped_mask = out_cropped_mask[crop_slice(crop)]
        if out_cropped_pts3d is not None:
            out_cropped_pts3d = out_cropped_pts3d[crop_slice(crop)]
        to_orig[:2, -1] = torch.tensor(crop[:2])

    return out_cropped_img, out_cropped_mask, out_cropped_pts3d, to_orig


@torch.no_grad()
def coarse_matching(
    query_view: dict,
    map_view: dict,
    model: AsymmetricMASt3R,
    device: torch.device,
    pixel_tol: int,
    fast_nn_params: dict,
    use_amp: bool = True,
    subsample: int = 8,
):
    # prepare batch
    imgs = []
    for idx, img in enumerate([query_view["rgb_rescaled"], map_view["rgb_rescaled"]]):
        imgs.append(
            dict(
                img=img.unsqueeze(0),
                true_shape=np.int32([img.shape[1:]]),
                idx=idx,
                instance=str(idx),
            )
        )
    output: dict = inference_fixed(
        [tuple(imgs)], model, device, batch_size=1, use_amp=use_amp, verbose=False
    )  # type: ignore
    pred1, pred2 = output["pred1"], output["pred2"]
    conf_list = [
        pred1["desc_conf"].squeeze(0).cpu().numpy(),
        pred2["desc_conf"].squeeze(0).cpu().numpy(),
    ]
    desc_list = [pred1["desc"].squeeze(0).detach(), pred2["desc"].squeeze(0).detach()]

    # find 2D-2D matches between the two images
    PQ, PM = desc_list[0], desc_list[1]
    if len(PQ) == 0 or len(PM) == 0:
        return [], [], [], []

    if pixel_tol == 0:
        matches_im_map, matches_im_query = cast(
            tuple[torch.Tensor, torch.Tensor],
            fast_reciprocal_NNs(PM, PQ, subsample_or_initxy1=8, **fast_nn_params),
        )
        HM, WM = map_view["rgb_rescaled"].shape[1:]
        HQ, WQ = query_view["rgb_rescaled"].shape[1:]
        # Ignore small border around the edge
        valid_matches_map = (
            (matches_im_map[:, 0] >= 3)
            & (matches_im_map[:, 0] < WM - 3)
            & (matches_im_map[:, 1] >= 3)
            & (matches_im_map[:, 1] < HM - 3)
        )
        valid_matches_query = (
            (matches_im_query[:, 0] >= 3)
            & (matches_im_query[:, 0] < WQ - 3)
            & (matches_im_query[:, 1] >= 3)
            & (matches_im_query[:, 1] < HQ - 3)
        )
        valid_matches = valid_matches_map & valid_matches_query
        matches_im_map = matches_im_map[valid_matches]
        matches_im_query = matches_im_query[valid_matches]
        valid_pts3d = []
        matches_confs = []
    else:
        if True:
            S = subsample
            HM, WM = map_view["rgb_rescaled"].shape[1:]
            yM, xM = np.mgrid[S // 2 : HM : S, S // 2 : WM : S].reshape(2, -1)
        else:
            yM, xM = torch.where(map_view["valid_rescaled"])
        matches_im_map, matches_im_query = cast(
            tuple[torch.Tensor, torch.Tensor],
            fast_reciprocal_NNs(
                PM,
                PQ,
                (xM, yM),  # type: ignore
                pixel_tol=pixel_tol,
                **fast_nn_params,
            ),
        )
        # valid_pts3d = (
        #     map_view["pts3d_rescaled"]
        #     .cpu()
        #     .numpy()[matches_im_map[:, 1], matches_im_map[:, 0]]
        # )
        valid_pts3d = []
        matches_confs = np.minimum(
            conf_list[1][matches_im_map[:, 1], matches_im_map[:, 0]],
            conf_list[0][matches_im_query[:, 1], matches_im_query[:, 0]],
        )

    # From cv2 to colmap
    matches_im_query = matches_im_query.astype(np.float64)
    matches_im_map = matches_im_map.astype(np.float64)
    matches_im_query[:, 0] += 0.5
    matches_im_query[:, 1] += 0.5
    matches_im_map[:, 0] += 0.5
    matches_im_map[:, 1] += 0.5
    # Rescale coordinates
    matches_im_query = geotrf(query_view["to_orig"], matches_im_query, norm=True)
    matches_im_map = geotrf(map_view["to_orig"], matches_im_map, norm=True)
    # From colmap back to cv2
    matches_im_query[:, 0] -= 0.5
    matches_im_query[:, 1] -= 0.5
    matches_im_map[:, 0] -= 0.5
    matches_im_map[:, 1] -= 0.5
    return valid_pts3d, matches_im_query, matches_im_map, matches_confs


@torch.no_grad()
def fine_matching(
    query_views: dict,
    map_views: dict,
    model: AsymmetricMASt3R,
    device: torch.device,
    max_batch_size: int,
    pixel_tol: int,
    fast_nn_params: dict,
    use_amp: bool = True,
    subsample: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert pixel_tol > 0
    output = crops_inference(
        [query_views, map_views],
        model,
        device,
        batch_size=max_batch_size,
        verbose=False,
    )
    pred1, pred2 = output["pred1"], output["pred2"]
    descs1 = pred1["desc"].clone()
    descs2 = pred2["desc"].clone()
    confs1 = pred1["desc_conf"].clone()
    confs2 = pred2["desc_conf"].clone()

    # Compute matches
    valid_pts3d, matches_im_map, matches_im_query, matches_confs = [], [], [], []
    for ppi, (pp1, pp2, cc11, cc21) in enumerate(zip(descs1, descs2, confs1, confs2)):
        # valid_ppi = map_views["valid"][ppi]
        # pts3d_ppi = map_views["pts3d"][ppi].cpu().numpy()
        conf_list_ppi = [cc11.cpu().numpy(), cc21.cpu().numpy()]

        S = subsample
        HM, WM = map_views["img"].shape[-2:]
        y_ppi, x_ppi = np.mgrid[S // 2 : HM : S, S // 2 : WM : S].reshape(2, -1)

        # y_ppi, x_ppi = torch.where(valid_ppi)
        matches_im_map_ppi, matches_im_query_ppi = fast_reciprocal_NNs(
            pp2, pp1, (x_ppi, y_ppi), pixel_tol=pixel_tol, **fast_nn_params
        )

        # valid_pts3d_ppi = pts3d_ppi[matches_im_map_ppi[:, 1], matches_im_map_ppi[:, 0]]
        matches_confs_ppi = np.minimum(
            conf_list_ppi[1][matches_im_map_ppi[:, 1], matches_im_map_ppi[:, 0]],
            conf_list_ppi[0][matches_im_query_ppi[:, 1], matches_im_query_ppi[:, 0]],
        )
        # inverse operation where we uncrop pixel coordinates
        matches_im_map_ppi = geotrf(
            map_views["to_orig"][ppi].cpu().numpy(),
            matches_im_map_ppi.copy(),
            norm=True,
        )
        matches_im_query_ppi = geotrf(
            query_views["to_orig"][ppi].cpu().numpy(),
            matches_im_query_ppi.copy(),
            norm=True,
        )

        matches_im_map.append(matches_im_map_ppi)
        matches_im_query.append(matches_im_query_ppi)
        # valid_pts3d.append(valid_pts3d_ppi)
        valid_pts3d.append(None)
        matches_confs.append(matches_confs_ppi)

    if len(valid_pts3d) == 0:
        return [], [], [], []

    matches_im_map = np.concatenate(matches_im_map, axis=0)
    matches_im_query = np.concatenate(matches_im_query, axis=0)
    # valid_pts3d = np.concatenate(valid_pts3d, axis=0)
    valid_pts3d = []
    matches_confs = np.concatenate(matches_confs, axis=0)
    return valid_pts3d, matches_im_query, matches_im_map, matches_confs


@torch.no_grad()
def crops_inference(
    pairs, model, device, batch_size=48, use_amp: bool = True, verbose=True
):
    assert len(pairs) == 2, (
        "Error, data should be a tuple of dicts containing the batch of image pairs"
    )
    # Forward a possibly big bunch of data, by blocks of batch_size
    B = pairs[0]["img"].shape[0]
    if B < batch_size:
        return loss_of_one_batch(
            pairs,
            model,
            None,
            device=device,
            symmetrize_batch=False,
            use_amp=use_amp,
        )
    preds = []
    for ii in range(0, B, batch_size):
        sel = slice(ii, ii + min(B - ii, batch_size))
        temp_data = [{}, {}]
        for di in [0, 1]:
            temp_data[di] = {
                kk: pairs[di][kk][sel]
                for kk in pairs[di].keys()
                if pairs[di][kk] is not None
            }  # copy chunk for forward
        preds.append(
            loss_of_one_batch(
                temp_data,
                model,
                None,
                device=device,
                symmetrize_batch=False,
                use_amp=use_amp,
            )
        )  # sequential forward
    # Merge all preds
    return cat_collate(preds, collate_fn_map=cat_collate_fn_map)


@torch.no_grad()
def inference_fixed(
    pairs, model, device, batch_size=8, use_amp: bool = True, verbose=True
):
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
