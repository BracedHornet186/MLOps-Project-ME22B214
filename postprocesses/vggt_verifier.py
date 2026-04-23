from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
import tqdm
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from colmap import remove_records
from scripts.data import resolve_model_path
from matchers.vggt import load_and_preprocess_images_imc25
from pipelines.scene import Scene
from postprocesses.base import TwoViewGeometryPruner
from postprocesses.config import VGGTTwoViewGeometryPrunerConfig
from storage import InMemoryKeypointStorage, InMemoryTwoViewGeometryStorage


class VGGTTwoViewGeometryPruner(TwoViewGeometryPruner):
    def __init__(
        self,
        conf: VGGTTwoViewGeometryPrunerConfig,
        device: torch.device,
    ):
        self.conf = conf
        self.device = device
        self.verifier = VGGTVerifier(
            resolve_model_path(conf.model.pretrained_model),
            device,
            threshold=self.conf.world_points_conf_threshold,
        )

    @torch.inference_mode()
    def __call__(
        self,
        scene: Scene,
        g_storage: InMemoryTwoViewGeometryStorage,
        keypoint_storage: InMemoryKeypointStorage | None = None,
        database_path: str = "colmap.db",
        progress_bar: tqdm.tqdm | None = None,
    ) -> InMemoryTwoViewGeometryStorage:
        assert keypoint_storage is not None
        pairs = []
        for key1 in g_storage.inliers.keys():
            for key2 in g_storage.inliers[key1].keys():
                idx1 = scene.short_key_to_idx(key1)
                idx2 = scene.short_key_to_idx(key2)
                pairs.append((idx1, idx2))

        pairs_to_remove = []
        for i, (idx1, idx2) in enumerate(pairs):
            if i >= self.conf.max_pairs:
                print(f"Break at {self.conf.max_pairs} pair")
                break

            path1 = scene.image_paths[idx1]
            path2 = scene.image_paths[idx2]

            try:
                kpts1 = keypoint_storage.get(path1)
                score = self.verifier.verify(path1, path2)

                if score < self.conf.score_threshold:
                    pairs_to_remove.append((idx1, idx2))
            except Exception as e:
                print(f"Verifier failed: {e}")

            if progress_bar:
                progress_bar.set_postfix_str(f"VGGT pruner ({i}/{len(pairs)})")

        pair_keys_to_remove = []
        for idx1, idx2 in pairs_to_remove:
            key1 = scene.idx_to_key(idx1)
            key2 = scene.idx_to_key(idx2)
            g_storage.remove(key1, key2)
            pair_keys_to_remove.append((key1, key2))

        print(f"# of pairs to remove: {len(pair_keys_to_remove)}")
        remove_records(pair_keys_to_remove, database_path=database_path)
        return g_storage


class VGGTVerifier:
    def __init__(
        self, weight_path: str | Path, device: torch.device, threshold: float = 1.5
    ):
        model = (
            VGGT.from_pretrained(resolve_model_path(str(weight_path))).eval().to(device)
        )
        self.model = model
        self.device = device
        self.threshold = threshold

        torch.cuda.synchronize()
        del self.model.point_head
        gc.collect()
        torch.cuda.empty_cache()
        self.model.point_head = None

    def verify(self, path1: str, path2: str, kpts1: np.ndarray):
        images, pads, resize_shapes, origin_shapes = load_and_preprocess_images_imc25(
            [path1, path2],
            mode="pad",
            target_size=518,
        )
        images = images.to(self.device, non_blocking=True)

        query_points = torch.from_numpy(kpts1).round().to(self.device)
        origin_h1, origin_w1 = origin_shapes[0]
        resize_h1, resize_w1 = resize_shapes[0]
        pad_left1, pad_top1 = pads[0]

        query_points[:, 0] = (query_points[:, 0] / origin_w1) * resize_w1 + pad_left1
        query_points[:, 1] = (query_points[:, 1] / origin_h1) * resize_h1 + pad_top1

        with torch.no_grad():
            with torch.autocast(self.device.type):
                predictions = self.model(images, query_points=query_points)

                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                    predictions["pose_enc"], images.shape[-2:]
                )
                pred_extrinsic = extrinsic[0]
                pred_intrinsic = intrinsic[0]

                # Get 3D points from depth map
                # You can also directly use the point map head to get 3D points, but its performance is slightly worse than the depth map
                depth_map, depth_conf = (
                    predictions["depth"][0],
                    predictions["depth_conf"][0],
                )
                world_points = unproject_depth_map_to_point_map(
                    depth_map, pred_extrinsic, pred_intrinsic
                )
                world_points = torch.from_numpy(world_points).to(self.device)
                world_points_conf = depth_conf.to(self.device)

                query_world_points_conf = world_points_conf[
                    query_points[:, 1], query_points[:, 0]
                ]

                # filtered_flag = pred_world_points_conf > 1.5
        pad_left, pad_top = pads[0]
        pad = max(pad_left, pad_top)
        pad_area = pad * 518

        score = (query_world_points_conf > self.threshold).sum() / (
            (518 * 518) - pad_area
        )
        return score
