from pathlib import Path
from typing import Any, Callable, Optional
import copy

import shutil
import numpy as np
import torch
import tqdm

from dust3r.cloud_opt.init_im_poses import align_multiple_poses

from data import FilePath
from localizers.base import PostLocalizer, Localizer
from localizers.common import get_default_K
from localizers.config import MapFreeConfig
from matchers.factory import create_detector_free_matcher
from models.mapfree.pose_solver import PnPSolver
from pipelines.scene import Scene
from storage import (
    InMemoryKeypointStorage,
    InMemoryMatchedKeypointStorage,
    InMemoryMatchingStorage,
)
from utils.camvis import save_camera_debug_info


class MapFreeFeatureMatching:
    def __init__(self, conf: MapFreeConfig, device: torch.device):
        self.conf = conf
        self.matcher = create_detector_free_matcher(conf.matcher, device=device)
        self.pose_solver = PnPSolver()
        print(f"[MapFree] Use matcher: {self.matcher}")

    @torch.inference_mode()
    def match(
        self, path1: FilePath, path2: FilePath, image_reader: Callable
    ) -> tuple[np.ndarray, np.ndarray]:
        matched_keypoint_storage = InMemoryMatchedKeypointStorage()
        self.matcher(path1, path2, matched_keypoint_storage, image_reader=image_reader)
        if not matched_keypoint_storage.has(path1, path2):
            return np.empty((0, 2)), np.empty((0, 2))
        mkpts1, mkpts2 = matched_keypoint_storage.get(path1, path2)
        return mkpts1, mkpts2

    def estimate_pose(
        self,
        mkpts1: np.ndarray,
        mkpts2: np.ndarray,
        depth1: np.ndarray,
        depth2: np.ndarray,
        K1: np.ndarray,
        K2: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        data = {
            "depth0": torch.from_numpy(depth1),
            "depth1": torch.from_numpy(depth2),
            "K_color0": torch.from_numpy(K1[None]),
            "K_color1": torch.from_numpy(K2[None]),
        }
        R, t, inliers = self.pose_solver.estimate_pose(mkpts1, mkpts2, data)
        R = torch.from_numpy(R.copy()).unsqueeze(0).float()
        t = torch.from_numpy(t.copy()).view(1, 3).unsqueeze(0).float()
        # world to cam

        return R, t, inliers


class MapFreeLocalizer(Localizer):
    def __init__(self, conf: MapFreeConfig, device: torch.device) -> None:
        self.device = device
        self.mapfree = MapFreeFeatureMatching(conf, device=self.device)
    
    @torch.inference_mode()
    def localize(
        self,
        scene: Scene,
        pairs: list[tuple[int, int]]
    ) -> dict:
        poses = {}
        for pair in pairs:
            i, j = pair
            print(f"pair=({pair})")

            if len(poses) == 0:
                poses = {i: (np.eye(3), np.zeros(3))}
            
            if i is poses and j in poses:
                print(f' -> Skip pair: {pair}')
                continue

            if i not in poses and j not in poses:
                print(f' -> Skip pair: {pair}')
                continue

            if i in poses:
                qid, xid = (j, i)
            else:
                qid, xid = (i, j)

            qpath = scene.image_paths[qid]
            xpath = scene.image_paths[xid]

            mkpts1, mkpts2 = self.mapfree.match(
                qpath, xpath, image_reader=scene.get_image
            )
            if len(mkpts1) == 0 or len(mkpts2) == 0:
                continue
            depth1 = scene.get_depth_image(qpath)
            depth2 = scene.get_depth_image(xpath)
            shape1 = scene.get_image_shape(qpath)
            shape2 = scene.get_image_shape(xpath)
            K1 = get_default_K(shape1)
            K2 = get_default_K(shape2)

            assert depth1 is not None
            assert depth2 is not None

            h1, w1 = shape1
            h2, w2 = shape2
            mkpts1[:, 0] = np.clip(mkpts1[:, 0], 0, w1 - 1)
            mkpts1[:, 1] = np.clip(mkpts1[:, 1], 0, h1 - 1)
            mkpts2[:, 0] = np.clip(mkpts2[:, 0], 0, w2 - 1)
            mkpts2[:, 1] = np.clip(mkpts2[:, 1], 0, h2 - 1)

            #R, t, inliers = self.mapfree.estimate_pose(
            #    mkpts1, mkpts2, depth1, depth2, K1, K2
            #)
            R, t, inliers = self.mapfree.estimate_pose(
                mkpts2, mkpts1, depth2, depth1, K2, K1
            )
            R = R.cpu().numpy()
            t = t.reshape(-1).cpu().numpy()
            # w2c?

            if np.isnan(R).any() or np.isnan(t).any() or np.isinf(t).any():
                continue

            Rx, tx = poses[xid]
            Rq = np.linalg.inv(R) @ Rx
            tq = -np.linalg.inv(R) @ t + tx

            poses[qid] = (Rq, tq)
    
        outputs = {}
        for i, pose in poses.items():
            name = Path(scene.image_paths[i]).name
            output_key = scene.data_schema.format_output_key(
                scene.dataset, scene.scene, name 
            )
            R, t = pose
            outputs[output_key] = {
                'R': copy.deepcopy(R),
                't': copy.deepcopy(t)
            }
        
        return outputs


class MapFreePostLocalizer(PostLocalizer):
    def __init__(self, conf: MapFreeConfig, device: torch.device) -> None:
        self.device = device
        self.mapfree = MapFreeFeatureMatching(conf, device=self.device)

    @torch.inference_mode()
    def localize(
        self,
        reference_sfm: Any,
        no_registered_query_output_keys: list[str],
        outputs: dict[str, dict[str, np.ndarray]],
        scene: Scene,
        keypoint_storage: InMemoryKeypointStorage,
        matching_storage: InMemoryMatchingStorage,
        progress_bar: Optional[tqdm.tqdm] = None
    ) -> dict:
        db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}

        rel_Rs = {}
        rel_ts = {}
        rel_xkeys = {}
        for i, qkey in enumerate(no_registered_query_output_keys):
            if progress_bar:
                progress_bar.set_postfix_str(
                    f"PostLocalizer ({i + 1}/{len(no_registered_query_output_keys)})"
                )
            qidx = scene.output_key_to_idx(qkey)
            qpath = scene.image_paths[qidx]
            qname = Path(qpath).name

            best_R = None
            best_t = None
            best_inliers = -1
            best_xkey = None
            for xkey in outputs.keys():
                xpath = scene.short_key_to_image_path(xkey)
                mkpts1, mkpts2 = self.mapfree.match(
                    qpath, xpath, image_reader=scene.get_image
                )
                if len(mkpts1) == 0 or len(mkpts2) == 0:
                    continue
                depth1 = scene.get_depth_image(qpath)
                depth2 = scene.get_depth_image(xpath)
                shape1 = scene.get_image_shape(qpath)
                shape2 = scene.get_image_shape(xpath)
                K1 = get_default_K(shape1)
                K2 = get_default_K(shape2)

                assert depth1 is not None
                assert depth2 is not None

                h1, w1 = shape1
                h2, w2 = shape2
                mkpts1[:, 0] = np.clip(mkpts1[:, 0], 0, w1 - 1)
                mkpts1[:, 1] = np.clip(mkpts1[:, 1], 0, h1 - 1)
                mkpts2[:, 0] = np.clip(mkpts2[:, 0], 0, w2 - 1)
                mkpts2[:, 1] = np.clip(mkpts2[:, 1], 0, h2 - 1)

                R, t, inliers = self.mapfree.estimate_pose(
                    mkpts1, mkpts2, depth1, depth2, K1, K2
                )
                R = R.cpu().numpy()
                t = t.reshape(-1).cpu().numpy()

                if np.isnan(R).any() or np.isnan(t).any() or np.isinf(t).any():
                    continue

                print(f'Inliers={inliers}')
                if inliers > best_inliers:
                    best_R = R
                    best_t = t
                    best_inliers = inliers
                    best_xkey = xkey

            if best_R is not None and best_t is not None:
                print(f'Best Inliers={best_inliers}')
                print('---------------')
                if best_inliers < 20:
                    continue
                rel_Rs[qkey] = best_R.copy()
                rel_ts[qkey] = best_t.copy()
                rel_xkeys[qkey] = best_xkey

            qR = rel_Rs.get(qkey)
            qt = rel_ts.get(qkey)
            xkey = rel_xkeys.get(qkey)
            if qR is None or qt is None or xkey is None:
                continue

            import roma
            from models.pixloc.utils.quaternions import qvec2rotmat
            trf = np.eye(4)

            #print(qR.shape)
            #qvec = roma.rotmat_to_unitquat(torch.from_numpy(qR[0]))[[3, 0, 1, 2]].numpy()
            #print(qvec.shape)
            #qR = qvec2rotmat(qvec)

            trf[:3, :3] = qR
            trf[:3, 3] = qt.ravel()

            Px = np.eye(4)
            Px[:3, :3] = outputs[xkey]['R']
            Px[:3, 3] = outputs[xkey]['t'].ravel()
            # c2w?

            Px[:3, :3] = np.linalg.inv(Px[:3, :3])
            Px[:3, 3] = -np.linalg.inv(Px[:3, :3]) @ Px[:3, 3]
            # w2c

            #s, R, T = align_multiple_poses(torch.from_numpy(np.linalg.inv(trf)[None]),
            #                               torch.from_numpy(Px[None]))
            #print(s, R, T)
            
            #Pq = np.linalg.inv(trf) @ Px
            #Pq = trf @ Px

            #Pq = np.eye(4)
            #Pq[:3, :3] = np.linalg.inv(trf[:3, :3]) @ Px[:3, :3]
            #Pq[:3, 3] = -np.linalg.inv(trf[:3, :3]) @ Px[:3, 3]
            Pq = np.linalg.inv(trf) @ Px

            #qvec = mat2quat(Pq[:3, :3])
            #tvec = Pq[:3, 3].ravel()
            outputs[qkey] = {
                'R': Pq[:3, :3].copy(),
                't': Pq[:3, 3].ravel().copy()
            }
            print(outputs[qkey])
            print(outputs[xkey])
            print('---')

            pose_debug_group = {
                qkey: outputs[qkey],
                xkey: outputs[xkey]
            }
            save_camera_debug_info(
                pose_debug_group,
                scene,
                Path('extra/cam_debug/mapfree_inv/') / str(qidx)
            )


def to_trf(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    trf = np.eye(4)
    trf[:3, :3] = R
    trf[:3, 3] = t
    return trf


def mat2quat(M):
    ''' Calculate quaternion corresponding to given rotation matrix

    Method claimed to be robust to numerical errors in `M`.

    Constructs quaternion by calculating maximum eigenvector for matrix
    ``K`` (constructed from input `M`).  Although this is not tested, a maximum
    eigenvalue of 1 corresponds to a valid rotation.

    A quaternion ``q*-1`` corresponds to the same rotation as ``q``; thus the
    sign of the reconstructed quaternion is arbitrary, and we return
    quaternions with positive w (q[0]).

    See notes.

    Parameters
    ----------
    M : array-like
      3x3 rotation matrix

    Returns
    -------
    q : (4,) array
      closest quaternion to input matrix, having positive q[0]

    References
    ----------
    * http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    * Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
      quaternion from a rotation matrix", AIAA Journal of Guidance,
      Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
      0731-5090

    Examples
    --------
    >>> import numpy as np
    >>> q = mat2quat(np.eye(3)) # Identity rotation
    >>> np.allclose(q, [1, 0, 0, 0])
    True
    >>> q = mat2quat(np.diag([1, -1, -1]))
    >>> np.allclose(q, [0, 1, 0, 0]) # 180 degree rotn around axis 0
    True

    Notes
    -----
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
    quaternion from a rotation matrix", AIAA Journal of Guidance,
    Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
    0731-5090

    '''
    # Qyx refers to the contribution of the y input vector component to
    # the x output vector component.  Qyx is therefore the same as
    # M[0,1].  The notation is from the Wikipedia article.
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    # Fill only lower half of symmetric matrix
    K = np.array([
        [Qxx - Qyy - Qzz, 0,               0,               0              ],
        [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
        [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
        ) / 3.0
    # Use Hermitian eigenvectors, values for speed
    vals, vecs = np.linalg.eigh(K)
    # Select largest eigenvector, reorder to w,x,y,z quaternion
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    # Prefer quaternion with positive w
    # (q * -1 corresponds to same rotation as q)
    if q[0] < 0:
        q *= -1
    return q