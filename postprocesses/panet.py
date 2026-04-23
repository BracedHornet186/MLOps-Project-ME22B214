import cv2
import torch
import numpy as np
import tqdm
from scripts.data import resolve_model_path, FilePath
from models.panet.model import PANet
from models.panet.refinement import refine_matches_coarse_to_fine
from pipelines.scene import Scene
from postprocesses.config import PANetRefinerConfig
from storage import InMemoryKeypointStorage, InMemoryMatchingStorage
from typing import Optional, Tuple


class PANetRefiner:
    def __init__(self, conf: PANetRefinerConfig,
                 device: Optional[torch.device] = None):
        weight_path = resolve_model_path(conf.weight_path)
        model = PANet(model_path=str(weight_path))
        self.conf = conf
        self.device = device
        self.model = model.eval().to(device)
    
    @torch.inference_mode()
    def refine_matched_keypoints(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        kpts1: np.ndarray,
        kpts2: np.ndarray,
        idxs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        resized_img1, fact1 = preprocess_image(
            img1, self.conf.max_edge, self.conf.max_sum_edges)
        resized_img2, fact2 = preprocess_image(
            img2, self.conf.max_edge, self.conf.max_sum_edges)
        
        # Downscale keypoints
        kpts1 *= 1 / fact1
        kpts2 *= 1 / fact2

        displacements12, displacements21 = refine_matches_coarse_to_fine(
            resized_img1, kpts1,
            resized_img2, kpts2,
            idxs,
            self.model, self.device, self.conf.batch_size, symmetric=True, grid=False
        )

        # NOTE:
        # displacements12: Shape(#idxs, 2), YX order
        # displacements21: Shape(#idxs, 2), YX order

        dx2 = displacements12[:, 1]
        dy2 = displacements12[:, 0]
        kpts2[idxs[:, 1], 0] += dx2 * 16
        kpts2[idxs[:, 1], 1] += dy2 * 16

        dx1 = displacements21[:, 1]
        dy1 = displacements21[:, 0]
        kpts1[idxs[:, 0], 0] += dx1 * 16
        kpts1[idxs[:, 0], 1] += dy1 * 16

        kpts1 *= fact1
        kpts2 *= fact2

        return kpts1, kpts2
    
    @torch.inference_mode()
    def refine_all(self,
                   scene: Scene,
                   keypoint_storage: InMemoryKeypointStorage,
                   matching_storage: InMemoryMatchingStorage,
                   progress_bar: Optional[tqdm.tqdm] = None):
        num_pairs = 0
        for key1, group in matching_storage:
            for key2 in group.keys():
                num_pairs += 1

        count = 0
        for key1, group in matching_storage:
            path1 = scene.image_paths[scene.short_key_to_idx(key1)]
            img1 = scene.get_image(str(path1))
            for key2, idxs in group.items():
                count += 1
                path2 = scene.image_paths[scene.short_key_to_idx(key2)]
                img2 = scene.get_image(str(path2))
                kpts1 = keypoint_storage.get(path1)
                kpts2 = keypoint_storage.get(path2)
                idxs = matching_storage.get(path1, path2)
                new_kpts1, new_kpts2 = self.refine_matched_keypoints(
                    img1.copy(), img2, kpts1, kpts2, idxs
                )
                keypoint_storage.keypoints[key1] = new_kpts1
                keypoint_storage.keypoints[key2] = new_kpts2
                if progress_bar:
                    progress_bar.set_postfix_str(
                        f'Refinement ({count}/{num_pairs})'
                    )


def preprocess_image(
    img: np.ndarray,
    max_edge: int,
    max_sum_edges: int
) -> Tuple[np.ndarray, float]:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fact = max(
        1,
        max(img.shape) / max_edge,
        sum(img.shape[: -1]) / max_sum_edges
    )
    resized_img = cv2.resize(img, None, fx=(1 / fact), fy=(1 / fact), interpolation=cv2.INTER_AREA)
    return resized_img, fact
        