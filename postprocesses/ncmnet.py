from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import tqdm

from scripts.data import resolve_model_path
from models.ncmnet.main import load_model
from models.ncmnet.ncmnet import NCMNet
from pipelines.common import Scene
from postprocesses.base import MatchingFilter
from postprocesses.config import NCMNetConfig
from scripts.storage import InMemoryKeypointStorage, InMemoryMatchingStorage


def read_image(path: str) -> np.ndarray:
    return cv2.imread(str(path))


class NCMNetFilter(MatchingFilter):
    def __init__(self,
                 conf: NCMNetConfig,
                 device: Optional[torch.device] = None):
        weight_path = resolve_model_path(conf.weight_path)
        model: NCMNet = load_model(str(weight_path))
        model = model.eval().to(device)

        self.conf = conf
        self.device = device
        self.model = model

    @torch.inference_mode()
    def run(self,
            keypoint_storage: InMemoryKeypointStorage,
            matching_storage: InMemoryMatchingStorage,
            scene: Scene,
            progress_bar: Optional[tqdm.tqdm] = None):
        """Filter matches and update the matching table
        """
        count = 0
        for path in scene.image_paths:
            k1 = Path(path).name
            if k1 not in matching_storage.matches:
                continue

            H1, W1 = scene.get_image_shape(str(path))
            for k2 in matching_storage.matches[k1].keys():
                count += 1

                H2, W2 = scene.get_image_shape(str(Path(scene.image_dir) / k2))

                idxs = matching_storage.matches[k1][k2].copy()
                kpts1 = keypoint_storage.get(k1).copy()
                kpts2 = keypoint_storage.get(k2).copy()
                mkpts1 = kpts1[idxs[:, 0]]
                mkpts2 = kpts2[idxs[:, 1]]

                if self.conf.topk and len(mkpts1) <= self.conf.topk:
                    # Nothing to do
                    print(f'[NCMNet] Skip by topk({self.conf.topk}) > {len(mkpts1)}')
                    continue

                inlier_masks, _ = self.filter_matches(
                    mkpts1, mkpts2, (H1, W1), (H2, W2)
                )
                if inlier_masks is None:
                    # NOTE: Remove current idxs?
                    continue

                inlier_idxs = idxs[inlier_masks, ...]
                print(f'[NCMNet] {inlier_idxs.shape}')
                matching_storage.matches[k1][k2] = inlier_idxs
                if progress_bar:
                    progress_bar.set_postfix_str(f'NCMNet filtering ({count})')
    
    @torch.inference_mode()
    def filter_matches(self,
                       mkpts1: np.ndarray,
                       mkpts2: np.ndarray,
                       shape1: Tuple[int, int],
                       shape2: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Filter matches

        Args
        ----
        mkpts1 : np.ndarray
        mkpts2 : np.ndarray
            Matched keypoints
        
        shape1 : Tuple[int, int]
        shape2 : Tuple[int, int]
            Size of an image (H, W)
        
        Returns
        -------
        inlier_masks: np.ndarray
            Inlier masking indices of shape(N,)

        inlier_probs: np.ndarray
            Inlier probs of shape(N,)
        """
        H1, W1 = shape1
        H2, W2 = shape2
        mkpts1 = np.clip(mkpts1 / np.array([W1, H1], dtype=np.float32), 0.0, 1.0)
        mkpts2 = np.clip(mkpts2 / np.array([W2, H2], dtype=np.float32), 0.0, 1.0)

        x1 = torch.from_numpy(mkpts1)[None][None].float().to(self.device, non_blocking=True)
        x2 = torch.from_numpy(mkpts2)[None][None].float().to(self.device, non_blocking=True)

        # NOTE:
        # x: Shape(1, 1, N, 4)
        # probs: Shape(1, N)
        x = torch.cat([x1, x2], dim=-1)
        _, _, _, outlier_scores = self.model(x, None)

        outlier_scores: np.ndarray = outlier_scores[0].cpu().numpy()    # Shape(N,)
        if len(outlier_scores) == 0:
            return None, None

        # NOTE:
        # The "scores" means outlier scores here (BUG?)
        #   (inlier) 0.0 <---------> inf (outlier)
        
        # NOTE:
        # Inverse and normalize "outlier_scores" to "inlier_probs" for convenience
        normalized_outlier_scores = (outlier_scores - outlier_scores.min()) / (outlier_scores.max() - outlier_scores.min())
        inlier_probs = 1.0 - normalized_outlier_scores

        if self.conf.inlier_ratio is not None:
            assert self.conf.inlier_prob_threshold is None
            assert self.conf.topk is None
            # Ratio-based filtering
            num_top = int(len(inlier_probs) * self.conf.inlier_ratio)
            num_top = max(1, num_top)
            th = np.sort(outlier_scores)[::1][num_top]
            inlier_masks = outlier_scores < th
        elif self.conf.inlier_prob_threshold is not None:
            assert self.conf.inlier_ratio is None
            assert self.conf.topk is None
            # Abs-threshold-based filtering
            inlier_masks = inlier_probs >= self.conf.inlier_prob_threshold
        elif self.conf.topk is not None:
            assert self.conf.inlier_prob_threshold is None
            assert self.conf.inlier_ratio is None
            th = np.sort(-inlier_probs)[self.conf.topk]
            th = th * (-1)
            inlier_masks = inlier_probs >= th
        else:
            raise ValueError

        return inlier_masks, inlier_probs
