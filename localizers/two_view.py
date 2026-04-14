import copy
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pycolmap
import scipy.spatial.distance
import torch
import tqdm

from data import FilePath
from localizers.base import Localizer, PostLocalizer
from localizers.common import get_default_K
from localizers.config import TwoViewLocalizerConfig
from pipelines.verification import run_ransac
from matchers.factory import create_detector_free_matcher
from pipelines.scene import Scene
from storage import (
    InMemoryKeypointStorage,
    InMemoryMatchedKeypointStorage,
    InMemoryMatchingStorage,
)
from utils.camvis import save_camera_debug_info


class TwoViewPostLocalizer(PostLocalizer):
    def __init__(self, conf: TwoViewLocalizerConfig, device: torch.device):
        self.conf = conf
        self.device = device
        self.matcher = create_detector_free_matcher(conf.matcher, device=device)
        print(f"[TwoViewPostLocalizer] Use matcher: {self.matcher}")

    @torch.inference_mode()
    def localize(
        self,
        reference_sfm: Any,
        no_registered_query_output_keys: list[str],
        outputs: dict[str, dict[str, np.ndarray]],
        scene: Scene,
        keypoint_storage: InMemoryKeypointStorage,
        matching_storage: InMemoryMatchingStorage,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> dict:
        threshold: float = 4.0
        db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}

        for i, qkey in enumerate(no_registered_query_output_keys):
            if progress_bar:
                progress_bar.set_postfix_str(
                    f"PostLocalizer ({i + 1}/{len(no_registered_query_output_keys)})"
                )
            qidx = scene.output_key_to_idx(qkey)
            qpath = scene.image_paths[qidx]
            qname = Path(qpath).name
            q_height, q_width = scene.get_image_shape(qpath)

            try:
                estimated_poses = []
                for xkey in outputs.keys():
                    xidx = scene.output_key_to_idx(xkey)
                    xpath = scene.image_paths[xidx]
                    # xkpts = keypoint_storage.get(xpath)
                    db_id = db_name_to_id.get(Path(xpath).name)
                    if db_id is None:
                        print(f"{Path(xpath).name} is not in DB. outputs[{xkey}]={outputs[xkey]}")
                        continue

                    ximage = reference_sfm.images[db_id]
                    if ximage.num_points3D == 0:
                        print(f"No 3D points found for {ximage.name}.")
                        continue

                    points3D_ids = np.array(
                        [p.point3D_id for p in ximage.points2D if p.has_point3D()]
                    )
                    xkpts = np.array([kp.xy for kp in ximage.points2D if kp.has_point3D()])
                    xkpts3d = np.array(
                        [reference_sfm.points3D[p3d_id].xyz for p3d_id in points3D_ids]
                    )
                    assert len(points3D_ids) == len(xkpts) == len(xkpts3d)
                    if len(points3D_ids) == 0:
                        continue

                    mkpt_storage = InMemoryMatchedKeypointStorage()
                    self.matcher(qpath, xpath, mkpt_storage, image_reader=scene.get_image)
                    if not mkpt_storage.has(qpath, xpath):
                        continue
                    mkpts1, mkpts2 = mkpt_storage.get(qpath, xpath)
                    print(f'{mkpts1.shape}, {mkpts2.shape}')
                    _, inliers = run_ransac(mkpts1, mkpts2, self.conf.ransac)
                    if len(inliers) == 0:
                        continue
                    inlier_mask = (inliers > 0).reshape(-1)
                    mkpts1 = mkpts1[inlier_mask]
                    mkpts2 = mkpts2[inlier_mask]
                    print(f'-> {mkpts1.shape}, {mkpts2.shape}')

                    # Distance between mkpts2 and pre-computed keypoints in image(x)
                    dists = scipy.spatial.distance.cdist(mkpts2, xkpts)
                    ranks = np.argsort(dists)
                    topk_ranks = ranks[:, :10]
                    topk_dists = np.take_along_axis(dists, topk_ranks, axis=1)

                    nearest_ranks = topk_ranks[:, 0]
                    nearest_dists = topk_dists[:, 0]

                    assert len(mkpts2) == len(topk_ranks) == len(topk_dists)

                    points2d = []
                    points3d = []
                    for kp1, rank, dist in zip(mkpts1, nearest_ranks, nearest_dists):
                        if dist > threshold:
                            continue
                        points2d.append(kp1)
                        points3d.append(xkpts3d[rank])

                    points2d = np.array(points2d)
                    points3d = np.array(points3d)
                    if len(points2d) == 0 or len(points3d) == 0:
                        continue

                    max_size = max((q_height, q_width))
                    FOCAL_PRIOR = 1.2
                    f = FOCAL_PRIOR * max_size
                    cam = pycolmap.Camera(
                        model="SIMPLE_RADIAL",
                        width=int(q_width),
                        height=int(q_height),
                        params=np.array([f, q_width / 2, q_height / 2, 0.1]),
                    )
                    print(f"-> -> {points2d.shape}, {points3d.shape}")
                    ret = pycolmap.absolute_pose_estimation(
                        points2d,
                        points3d,
                        cam,
                        estimation_options={"ransac": {"max_error": 12.0}},
                        refinement_options={},
                    )

                    if ret is None:
                        print(f"[TwoViewPostLocalizer] ({qkey}, {xkey}): Faild to estimate a pose")
                    else:
                        num_inliers = ret["num_inliers"]
                        cam_from_world = ret["cam_from_world"]
                        estimated_poses.append((cam_from_world, num_inliers))
                        print(f"[TwoViewPostLocalizer] ({qkey}, {xkey}): #inliers={num_inliers}")
            except Exception as e:
                print(f"[TwoViewPostLocalizer] Error: {e} ({qkey})")

            if len(estimated_poses) == 0:
                print(f"[TwoViewPostLocalizer] {qkey} cannot be localized")
                continue

            best_pose = list(
                sorted(estimated_poses, key=lambda pose: pose[1], reverse=True)
            )[0]
            cam_from_world, num_inliers = best_pose
            key1 = scene.data_schema.format_output_key(
                scene.dataset, scene.scene, qname
            )
            outputs[key1] = {
                "R": copy.deepcopy(cam_from_world.rotation.matrix()),
                "t": copy.deepcopy(np.array(cam_from_world.translation)),
            }
            print(
                f"[TwoViewPostLocalizer] {key1} was localized (# of inliers: {num_inliers})"
            )
        return outputs
