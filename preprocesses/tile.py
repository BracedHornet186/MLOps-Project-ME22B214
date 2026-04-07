import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def extract_patches_2d_with_overlap(
    img: torch.Tensor, size: tuple[int, int]
) -> tuple[torch.Tensor, list[dict[str, int]]]:
    """https://discuss.pytorch.org/t/how-to-extract-smaller-image-patches-3d/16837/8

    Args
    ----
    img : torch.Tensor
        Shape(1, 3, H, W)

    size : tuple[int, int]
        Patch size (Height, Width)

    Returns
    -------
    torch.Tensor, list[dict[str, int]]
        - Tiled images, Shape(# of tiles, 3, size[0], size[1])
        - Position info for each patch
    """
    patch_H, patch_W = get_real_patch_size(img, size)
    patches_fold_H = img.unfold(2, patch_H, patch_H)
    if img.size(2) % patch_H != 0:
        patches_fold_H = torch.cat(
            (
                patches_fold_H,
                img[
                    :,
                    :,
                    -patch_H:,
                ]
                .permute(0, 1, 3, 2)
                .unsqueeze(2),
            ),
            dim=2,
        )
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, patch_W)
    if img.size(3) % patch_W != 0:
        patches_fold_HW = torch.cat(
            (
                patches_fold_HW,
                patches_fold_H[:, :, :, -patch_W:, :]
                .permute(0, 1, 2, 4, 3)
                .unsqueeze(3),
            ),
            dim=3,
        )
    patches = patches_fold_HW.permute(0, 2, 3, 1, 4, 5).reshape(
        -1, img.size(1), patch_H, patch_W
    )

    info_list = get_patch_position_info_list(img, size)
    return patches, info_list


def get_real_patch_size(
    img: torch.Tensor,
    size: tuple[int, int]
) -> tuple[int, int]:
    patch_H, patch_W = min(img.size(2), size[0]), min(img.size(3), size[1])
    return (patch_H, patch_W)


def get_patch_position_info_list(
    img: torch.Tensor,
    size: tuple[int, int],
) -> list[dict[str, int]]:
    patch_h, patch_w = get_real_patch_size(img, size)
    h_anchors, w_anchors = get_patch_anchors(img, size)
    hpos, wpos = np.meshgrid(h_anchors, w_anchors, indexing='ij')

    info_list = []
    for row_idx, (ys, xs) in enumerate(zip(hpos, wpos)):
        n_cols = len(ys)
        for col_idx, (y, x) in enumerate(zip(ys, xs)):
            patch_idx = row_idx * n_cols + col_idx

            left = x
            top = y
            right = x + patch_w
            bottom = y + patch_h

            shift_h = 0
            shift_w = 0

            if right > img.size(-1):
                shift_w = right - img.size(-1)
                left -= shift_w
                right -= shift_w

            if bottom > img.size(-2):
                shift_h = bottom - img.size(-2)
                top -= shift_h
                bottom -= shift_h
            
            info = {
                "index": patch_idx,     # Index of patch list
                "left": left,           # Left point on the original coordinate
                "top": top,             # Top point on the original coordinate
                "right": right,         # Right point on the original coordinate
                "bottom": bottom,       # Bottom point on the original coordinate
                "shift_h": shift_h,
                "shift_w": shift_w,
                "patch_h": patch_h,
                "patch_w": patch_w
            }
            info_list.append(info)
    return info_list


def get_patch_anchors(
    img: torch.Tensor, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args
    ----
    img : torch.Tensor
        Shape(1, 3, H, W)

    size : tuple[int, int]
        Patch size (Height, Width)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Anchor points of y-coordinates and x-coordinates
    """
    orig_H, orig_W = img.size(2), img.size(3)
    patch_H, patch_W = get_real_patch_size(img, size)
    return np.arange(0, orig_H, patch_H), np.arange(0, orig_W, patch_W)


def show_tile_images(
    tile_imgs: torch.Tensor, origin_size: tuple[int, int]
):
    _, origin_width = origin_size
    tile_width = tile_imgs.size(-2)
    nrow = int(
        np.ceil(float(origin_width) / tile_width)
    )  # Number of patches per row (= column size)
    show_patches = make_grid(tile_imgs, nrow=nrow).permute(1, 2, 0).numpy()
    plt.imshow(show_patches)
    plt.show()


def reconstruct_image(
    tile_imgs: torch.Tensor,
    position_info_list: list[dict[str, int]],
    origin_size: tuple[int, int]
):
    origin_h, origin_w = origin_size
    rec_img = torch.zeros((3, origin_h, origin_w)).to(tile_imgs.dtype)
    for tile_img, position_info in zip(tile_imgs, position_info_list):
        left = position_info["left"]
        top = position_info["top"]
        right = position_info["right"]
        bottom = position_info["bottom"]
        rec_img[:, top:bottom, left:right] = tile_img.clone()
    return rec_img
