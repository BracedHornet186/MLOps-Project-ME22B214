from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.feature.loftr.loftr import LoFTR
from PIL import Image

from data import resolve_model_path
from matchers.visualizer import draw_img
from models.minima.depth_model import init_depth_model, predict_depth
from models.minima.model import create_minima_lightglue, create_minima_loftr
from pipelines.verification import run_ransac
from postprocesses.config import MINIMAVerifierConfig


class MINIMAVerifier:
    def __init__(self, conf: MINIMAVerifierConfig, device: torch.device):
        self.conf = conf
        self.depth_model = init_depth_model(
            resolve_model_path(conf.depth_model_weight_path), device
        )
        if conf.matcher_type == "splg":
            assert conf.sp_weight_path
            assert conf.lg_weight_path
            self.matcher = create_minima_lightglue(
                resolve_model_path(conf.sp_weight_path),
                resolve_model_path(conf.lg_weight_path),
                device,
            )
        elif conf.matcher_type == "loftr":
            assert conf.loftr_weight_path
            self.matcher = create_minima_loftr(
                resolve_model_path(conf.loftr_weight_path),
                device,
            )
        else:
            raise ValueError
        self.device = device

    def requires_grayscale(self) -> bool:
        if self.conf.matcher_type == "loftr":
            return True
        return False

    def verify_rgb_depth_consistency(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        ax: plt.Axes | None = None,
    ) -> bool:
        mkpts1, mkpts2, _ = self.match_rgb_depth(img1, img2, ax=ax)
        try:
            _, inliers = run_ransac(mkpts1, mkpts2, self.conf.ransac)
        except Exception as e:
            print(f"MINIMAVerifier | RANSAC failed: {e}")
            return False
        inlier_mask = (inliers > 0).reshape(-1)
        num_inliers = sum(inlier_mask)
        return num_inliers >= self.conf.inlier_threshold

    def verify_depth_depth_consistency(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        ax: plt.Axes | None = None,
    ) -> bool:
        mkpts1, mkpts2, _ = self.match_depth_depth(img1, img2, ax=ax)
        try:
            _, inliers = run_ransac(mkpts1, mkpts2, self.conf.ransac)
        except Exception as e:
            print(f"MINIMAVerifier | RANSAC failed: {e}")
            return False
        inlier_mask = (inliers > 0).reshape(-1)
        num_inliers = sum(inlier_mask)
        return num_inliers >= self.conf.inlier_threshold

    @torch.inference_mode()
    def match_rgb_depth(
        self,
        img1: np.ndarray,  # (H, W, 3), BGR order, For RGB
        img2: np.ndarray,  # (H, W, 3), BGR order, For Depth
        ax: plt.Axes | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        img2_depth = predict_depth(img2, self.depth_model, self.device)
        tensor1, scale1, _ = preprocess_image(
            img1,
            gray_scale=self.requires_grayscale(),
        )
        tensor2, scale2, _ = preprocess_image(
            img2_depth,
            gray_scale=self.requires_grayscale(),
        )
        mkpts1, mkpts2, scores = self.match(tensor1, tensor2, scale1, scale2)
        if ax is not None:
            print(f"matches={len(mkpts1)}")
            draw_img(
                Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)),
                Image.fromarray(cv2.cvtColor(img2_depth, cv2.COLOR_BGR2RGB)),
                mkpts1,
                mkpts2,
                ax=ax,
            )
        return mkpts1, mkpts2, scores

    @torch.inference_mode()
    def match_depth_depth(
        self,
        img1: np.ndarray,  # (H, W, 3), BGR order, For Depth
        img2: np.ndarray,  # (H, W, 3), BGR order, For Depth
        ax: plt.Axes | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        img1_depth = predict_depth(img1, self.depth_model, self.device)
        img2_depth = predict_depth(img2, self.depth_model, self.device)
        tensor1, scale1, _ = preprocess_image(
            img1_depth,
            gray_scale=self.requires_grayscale(),
        )
        tensor2, scale2, _ = preprocess_image(
            img2_depth,
            gray_scale=self.requires_grayscale(),
        )
        mkpts1, mkpts2, scores = self.match(tensor1, tensor2, scale1, scale2)
        if ax is not None:
            draw_img(
                Image.fromarray(cv2.cvtColor(img1_depth, cv2.COLOR_BGR2RGB)),
                Image.fromarray(cv2.cvtColor(img2_depth, cv2.COLOR_BGR2RGB)),
                mkpts1,
                mkpts2,
                ax=ax,
            )
        return mkpts1, mkpts2, scores

    @torch.inference_mode()
    def match(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        scale1: np.ndarray,
        scale2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        img1 = img1.to(self.device, non_blocking=True)
        img2 = img2.to(self.device, non_blocking=True)
        batch = {"image0": img1, "image1": img2}
        pred = self.matcher(batch)

        if self.conf.matcher_type == "splg":
            mkpts1 = pred["keypoints0"].cpu().numpy()
            mkpts2 = pred["keypoints1"].cpu().numpy()
            matching_scores = pred["matching_scores"].detach().cpu().numpy()
            mkpts1 = mkpts1 * scale1
            mkpts2 = mkpts2 * scale2
        elif self.conf.matcher_type == "loftr":
            mkpts1 = pred["keypoints0"].cpu().numpy()
            mkpts2 = pred["keypoints1"].cpu().numpy()
            matching_scores = pred["confidence"].detach().cpu().numpy()
        else:
            raise ValueError(self.conf.matcher_type)
        return mkpts1, mkpts2, matching_scores


def preprocess_image(
    img: np.ndarray,
    resize: int = 640,
    df: int = 8,
    padding: bool = False,
    gray_scale: bool = False,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor | None]:
    # xoftr takes grayscale input images
    if gray_scale and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]

    if resize is not None:
        scale = resize / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    else:
        w_new, h_new = w, h

    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w_new, h_new])

    img = cv2.resize(img, (w_new, h_new))
    scale = np.array([w / w_new, h / h_new], dtype=float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        # TODO
        img, mask = self.pad_bottom_right(img, pad_to, ret_mask=True)
        mask = torch.from_numpy(mask)
    else:
        mask = None
    # img = transforms.functional.to_tensor(img).unsqueeze(0).to(device)
    if len(img.shape) == 2:  # grayscale image
        img = torch.from_numpy(img)[None][None].float() / 255.0
    else:  # Color image
        img = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
    return img, scale, mask
