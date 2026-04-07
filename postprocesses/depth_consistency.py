from __future__ import annotations

import cv2
import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
import tqdm
from mpsfm.sfm.scene.image.utils import get_continuity_mask
from mpsfm.utils.geometry import project3D, unproject_depth_map_to_world

from models.metric3d.model import Metric3DDepthModel
from pipelines.scene import Scene


class Depth:
    def __init__(
        self,
        depth_dict: dict[str, np.ndarray],
        camera: pycolmap.Camera,
        kps: np.ndarray,
        mask: np.ndarray | None = None,
        depth_uncertainty: float | None = 0.05,  # 0.0263,
        prior_uncertainty: bool = True,
        prior_std_multiplier: float = 3.7757,  # 3.33,
        inherent_noise: float = 0.02,
        max_std: float | None = None,
        std_multiplier: float = 1.0,
        depth_lim: float | None = None,
        **kwargs,
    ):
        self.kps = kps
        self.camera = camera

        self.scale = 1
        self.shift = 0
        self.activated = False

        self.depth_uncertainty = depth_uncertainty
        self.prior_uncertainty = prior_uncertainty
        self.prior_std_multiplier = prior_std_multiplier
        self.inherent_noise = inherent_noise
        self.max_std = max_std
        self.std_multiplier = std_multiplier
        self.depth_lim = depth_lim

        variances = []
        mews = []
        assert "depth_variance" in depth_dict, (
            "Variance must be provided for prior uncertainty"
        )
        mews.append(depth_dict["depth"])
        variances.append(depth_dict["depth_variance"])
        valid_mask = depth_dict["depth"] > 0
        valid_mask *= depth_dict["valid"]
        continuity_mask = get_continuity_mask(depth_dict["depth"])
        assert len(mews) == 1
        self.data_prior = mews[0]

        if self.depth_uncertainty is not None:
            new_var = []
            if self.prior_uncertainty:
                for mewi, vari in zip(mews, variances):
                    new_var.append(
                        np.maximum(
                            vari * self.prior_std_multiplier**2,
                            (mewi * self.depth_uncertainty) ** 2,
                        )
                    )
                if len(new_var) > 1:
                    self.uncertainty = 1 / (
                        np.sum(1 / (vari + 1e-6) for vari in new_var) + 1e-6
                    )
                else:
                    self.uncertainty = new_var[0]
            else:
                self.uncertainty = (self.data_prior * self.depth_uncertainty) ** 2
        else:
            assert len(variances) == 1, "Only one variance is supported for now"
            self.uncertainty = variances[0]

        self.uncertainty = self.uncertainty.clip(
            self.inherent_noise**2,
            self.max_std if self.max_std is None else self.max_std**2,
        )
        self.uncertainty = self.uncertainty * (self.std_multiplier**2)

        if mask is not None and self.uncertainty is not None:
            if mask.shape != self.uncertainty.shape:
                mask = (
                    cv2.resize(
                        mask.astype(np.float32),
                        self.uncertainty.shape[::-1],
                        interpolation=cv2.INTER_NEAREST,
                    )
                    > 0.5
                )
            valid_mask *= mask
        self.uncertainty[~valid_mask] = 1e6
        self.valid = valid_mask
        zero_depth_mask = self.data_prior == 0
        self.data_prior[zero_depth_mask] = 0.1
        self.valid[zero_depth_mask] = False

        if self.depth_lim is not None:
            self.valid[self.data_prior > self.depth_lim] = False

        self.uncertainty_update = data_at_kps(kps, self.uncertainty)
        self.data = self.data_prior.copy()


def data_at_kps(
    kps: np.ndarray, data: np.ndarray, mode: str = "bilinear"
) -> np.ndarray:
    kp_t = torch.tensor(kps, dtype=torch.float64)
    if len(kp_t.shape) == 1:
        kp_t = kp_t[None]
    H, W = data.shape[:2]
    data_map_t = torch.tensor(data, dtype=torch.float64)[None, None]
    kp_t[:, 0] = (kp_t[:, 0] / (W - 1)) * 2 - 1
    kp_t[:, 1] = (kp_t[:, 1] / (H - 1)) * 2 - 1
    kp_t = kp_t[None, None].permute(0, 1, 2, 3)
    sampled_data = F.grid_sample(
        data_map_t, kp_t, mode=mode, padding_mode="zeros", align_corners=True
    )[0, 0, 0].numpy()

    return sampled_data


def reproject_depth(
    image1: pycolmap.Image,
    image2: pycolmap.Image,
    depth1: np.ndarray,
    depth2_shape: tuple[int, ...],
    cam1: pycolmap.Camera,
    cam2: pycolmap.Camera,
    cfw1: pycolmap.Rigid3d | None = None,
    cfw2: pycolmap.Rigid3d | None = None,
) -> dict[str, np.ndarray]:
    """Reprojects depth from imid1 to imid2

    NOTE:
    Adapted from mpsfm/sfm/scene/reconstruction/mixins/depth_utils.py
    """
    # depth1 = self.images[imid1].depth.data
    # depth2_shape = self.images[imid2].depth.data_prior.shape

    valid1_mask = np.ones(depth1.shape, dtype=bool)
    depth1[depth1 <= 0] = 0.1
    if cfw1 is None:
        cfw1 = image1.cam_from_world
    if cfw2 is None:
        cfw2 = image2.cam_from_world

    assert isinstance(cfw1, pycolmap.Rigid3d)
    assert isinstance(cfw2, pycolmap.Rigid3d)
    H1 = np.concatenate([cfw1.matrix(), np.array([[0, 0, 0, 1]])], axis=0)
    H2 = np.concatenate([cfw2.matrix(), np.array([[0, 0, 0, 1]])], axis=0)
    K1 = cam1.calibration_matrix()

    sx = 1.0
    sy = 1.0

    K1[0, :] *= sx
    K1[1, :] *= sy
    D1W = unproject_depth_map_to_world(depth1, K1, np.linalg.inv(H1), mask=valid1_mask)
    K2 = cam2.calibration_matrix()
    K2[0, :] *= sx
    K2[1, :] *= sy
    p2D12, depth12 = project3D(D1W, H2, K2)

    mask12 = (
        (p2D12[:, 0] >= 0)
        & ((p2D12[:, 0] + 0.5) < depth2_shape[1])
        & (p2D12[:, 1] >= 0)
        & ((p2D12[:, 1] + 0.5) < depth2_shape[0])
    )
    mask12 *= depth12 > 0
    out = {
        "depth1": depth1,
        "p2D12": p2D12.reshape(*(depth1.shape), 2),
        "depth12": depth12.reshape(depth1.shape),
        "mask12": mask12.reshape(depth1.shape),
        "valid1_mask": valid1_mask,
        "points_3d": D1W,
    }

    return out


def check_depth_consistency(
    imid1: int,
    imid2: int,
    depth_dict: dict[int, Depth],
    rec: pycolmap.Reconstruction,
    c=15,
    score_thresh: float | None = None,
    depth_cons_valid_thresh: float = 0.6,
):
    """Check depth consistency between two images."""
    image1 = rec.images[imid1]
    image2 = rec.images[imid2]
    cam1 = image1.camera
    cam2 = image2.camera

    if cam1 is None or cam2 is None:
        return

    out = reproject_depth(
        image1, image2, depth_dict[imid1].data, depth_dict[imid2].data.shape, cam1, cam2
    )
    out21 = reproject_depth(
        image2, image1, depth_dict[imid2].data, depth_dict[imid1].data.shape, cam2, cam1
    )
    out |= {
        rev_key: out21[key]
        for key, rev_key in zip(
            ["depth1", "p2D12", "depth12", "mask12", "valid1_mask"],
            ["depth2", "p2D21", "depth21", "mask21", "valid2_mask"],
        )
    }

    isdepth1_in_canv = out["mask12"].reshape(out["valid1_mask"].shape)
    isdepth2_in_canv = out["mask21"].reshape(out["valid2_mask"].shape)

    p2D12 = out["p2D12"][out["mask12"]].astype(int)
    p2D21 = out["p2D21"][out["mask21"]].astype(int)

    p2D12 = out["p2D12"][isdepth1_in_canv].astype(int)
    p2D21 = out["p2D21"][isdepth2_in_canv].astype(int)
    minbuffer12, mask12buffer_ = find_min_buffer(
        out["depth12"][isdepth1_in_canv], p2D12, isdepth2_in_canv.shape
    )
    mask12buffer = np.zeros(isdepth1_in_canv.shape, dtype=bool)
    mask12buffer[isdepth1_in_canv] = mask12buffer_
    minbuffer21, mask21buffer_ = find_min_buffer(
        out["depth21"][isdepth2_in_canv], p2D21, isdepth1_in_canv.shape
    )
    mask21buffer = np.zeros(isdepth2_in_canv.shape, dtype=bool)
    mask21buffer[isdepth2_in_canv] = mask21buffer_
    if score_thresh is None:
        score_thresh = depth_cons_valid_thresh

    var1 = depth_dict[imid1].uncertainty.copy()
    var1 /= depth_dict[imid1].prior_std_multiplier ** 2
    y, x = np.where(mask12buffer)
    keypoints1 = np.array([x, y]).T
    covs1 = lifted_pointcovs_cam(
        out["depth1"][mask12buffer],
        cam1,
        keypoints1,
        var1[mask12buffer],
    )
    covs1w = rotate_covs_to_world(covs1, image1)
    covs12 = rotate_covs_to_cam(covs1w, image2)
    std1 = var1**0.5
    std1bar = covs12[:, 2, 2] ** 0.5

    var2 = depth_dict[imid2].uncertainty.copy()
    var2 /= depth_dict[imid2].prior_std_multiplier ** 2
    y, x = np.where(mask21buffer)
    keypoints2 = np.array([x, y]).T
    covs2 = lifted_pointcovs_cam(
        out["depth2"][mask21buffer],
        cam2,
        keypoints2,
        var2[mask21buffer],
    )
    covs2w = rotate_covs_to_world(covs2, image2)
    covs21 = rotate_covs_to_cam(covs2w, image1)
    std2 = var2**0.5
    std2bar = covs21[:, 2, 2] ** 0.5
    t1 = minbuffer12[p2D12[:, 1], p2D12[:, 0]] - out["depth2"][p2D12[:, 1], p2D12[:, 0]]
    t1 /= ((std1bar * c) ** 2 + (std2[p2D12[:, 1], p2D12[:, 0]] * c) ** 2) ** 0.5
    t2 = minbuffer21[p2D21[:, 1], p2D21[:, 0]] - out["depth1"][p2D21[:, 1], p2D21[:, 0]]
    t2 /= ((std2bar * c) ** 2 + (std1[p2D21[:, 1], p2D21[:, 0]] * c) ** 2) ** 0.5

    print(t1, t2)

    surface1 = np.abs(t1) < score_thresh
    occl1 = t1 > score_thresh
    invalid1 = t1 < -score_thresh
    surface2 = np.abs(t2) < score_thresh
    occl2 = t2 > score_thresh
    invalid2 = t2 < -score_thresh

    valid1 = np.zeros(isdepth1_in_canv.shape, dtype=bool)
    valid2 = np.zeros(isdepth2_in_canv.shape, dtype=bool)

    valid1[isdepth1_in_canv] = surface1 + occl1
    valid2[isdepth2_in_canv] = surface2 + occl2
    occl1_ = np.zeros(isdepth1_in_canv.shape, dtype=bool)
    occl2_ = np.zeros(isdepth2_in_canv.shape, dtype=bool)
    occl1_[isdepth1_in_canv] = occl1
    occl2_[isdepth2_in_canv] = occl2
    invalid1_ = np.zeros(isdepth1_in_canv.shape, dtype=bool)
    invalid2_ = np.zeros(isdepth2_in_canv.shape, dtype=bool)
    invalid1_[isdepth1_in_canv] = invalid1
    invalid2_[isdepth2_in_canv] = invalid2
    surface1_ = np.zeros(isdepth1_in_canv.shape, dtype=bool)
    surface2_ = np.zeros(isdepth2_in_canv.shape, dtype=bool)
    surface1_[isdepth1_in_canv] = surface1
    surface2_[isdepth2_in_canv] = surface2

    return {
        "valid1": valid1,
        "valid2": valid2,
        "occl1": occl1_,
        "occl2": occl2_,
        "invalid1": invalid1_,
        "invalid2": invalid2_,
        "surface1": surface1_,
        "surface2": surface2_,
        "valid1_mask": isdepth1_in_canv,
        "valid2_mask": isdepth2_in_canv,
    }


def find_min_buffer(
    Dab: np.ndarray, Pab: np.ndarray, buffer_shapes
) -> tuple[np.ndarray, np.ndarray]:
    """Find the minimum distance buffer for depth consistency check."""
    min_dabs_out = np.full(buffer_shapes, np.inf)
    Pab_x, Pab_y = np.array(Pab).T

    # Update minimum distances and corresponding indices
    mask = Dab < min_dabs_out[Pab_y, Pab_x]
    min_dabs_out[Pab_y[mask], Pab_x[mask]] = Dab[mask]

    return min_dabs_out, mask


def lifted_pointcovs_cam(
    dd: np.ndarray,
    camera: pycolmap.Camera,
    keypoints: np.ndarray,
    var_d: np.ndarray,
    sigma_q: float = 1,
) -> np.ndarray:
    """Computes the covariance of lifted points in camera coordinates."""
    cc = np.array([camera.principal_point_x, camera.principal_point_y])
    ff = np.array([camera.focal_length_x, camera.focal_length_y])
    ff_inv = 1.0 / ff

    dpdd = np.hstack(
        [(keypoints - cc[None, :]) * ff_inv[None, :], np.ones((keypoints.shape[0], 1))]
    )[:, :, None]

    dpdq = np.zeros((keypoints.shape[0], 2, 3))
    dpdq[:, 0, 0] = dd * ff_inv[0]
    dpdq[:, 1, 1] = dd * ff_inv[1]
    dpdq = np.clip(dpdq, -1e6, 1e6)

    Cov_d = var_d[:, None, None] * np.einsum("nij,nkj->nik", dpdd, dpdd)

    Cov_q_small = sigma_q**2 * np.einsum("nij,nkj->nik", dpdq, dpdq)
    Cov_q = np.zeros((dpdq.shape[0], 3, 3))
    Cov_q[:, :2, :2] = Cov_q_small

    Cov_plifted_cam = Cov_d + Cov_q

    return Cov_plifted_cam


def rotate_covs(Covs: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotates covariances by rotation matrix R"""
    return R[None] @ Covs @ R.T[None]


def rotate_covs_to_world(Covs: np.ndarray, image: pycolmap.Image):
    """Rotates covariances to world coordinates"""
    assert image.cam_from_world is not None
    R = image.cam_from_world.rotation.matrix()
    return rotate_covs(Covs, R)


def rotate_covs_to_cam(Covs_world: np.ndarray, image: pycolmap.Image):
    """Rotates covariances to camera coordinates from world coordinates"""
    assert image.cam_from_world is not None
    R = image.cam_from_world.rotation.matrix()
    return rotate_covs(Covs_world, R.T)


def compute_depth_concistency_score(
    imid: int,
    target_imids: list[int],
    depth_dict: dict[int, Depth],
    rec: pycolmap.Reconstruction,
    score_thresh: float | None = None,
):
    """Check depth consistency of a bundle of images."""
    optim_ids = list(set(target_imids) - {imid})
    collect = {}
    for ref_imid in optim_ids:
        collect[ref_imid] = check_depth_consistency(
            imid, ref_imid, depth_dict, rec, score_thresh=score_thresh
        )
    ref_notvalid = [~v["valid2"] * v["valid2_mask"] for _, v in collect.items()]
    ref_mask = [v["valid2_mask"] * ~v["occl2"] for _, v in collect.items()]
    qry_notvalid = [~v["valid1"] * v["valid1_mask"] for _, v in collect.items()]
    qry_mask = [v["valid1_mask"] * ~v["occl1"] for _, v in collect.items()]
    ref_sum_not_valids = [np.sum(notvalid) for notvalid in ref_notvalid]
    ref_sum_valids = [np.sum(valid) for valid in ref_mask]
    qry_sum_not_valids = [np.sum(notvalid) for notvalid in qry_notvalid]
    qry_sum_valids = [np.sum(valid) for valid in qry_mask]
    tot_ref_im_ratios = np.sum(ref_sum_not_valids) / (np.sum(ref_sum_valids).clip(0.1))
    tot_qry_im_ratios = np.sum([qry_sum_not_valids]) / (
        np.sum(qry_sum_valids).clip(0.1)
    )

    max_ratios = np.max([tot_ref_im_ratios, tot_qry_im_ratios])
    return max_ratios, (
        sum(np.sum(v["valid1_mask"]) for v in collect.values()),
        sum(np.sum(v["valid2_mask"]) for v in collect.values()),
    )


def extract_depth(
    model: Metric3DDepthModel,
    scene: Scene,
    rec: pycolmap.Reconstruction,
) -> dict[int, Depth]:
    depth_dict = {}
    for imid, image in tqdm.tqdm(
        rec.images.items(),
        total=len(rec.images),
        desc="Depth extraction",
    ):
        assert isinstance(image, pycolmap.Image)
        i = scene.short_key_to_idx(image.name)
        camera = image.camera
        if camera is None:
            continue

        intrinsics = camera.params.copy()

        path = scene.image_paths[i]
        img = scene.get_image(path)
        output = model.predict(img, intrinsics)

        kpts = np.array([kp.xy for kp in image.points2D], dtype=np.float32)
        depth = Depth(
            {k: v.squeeze().cpu().numpy() for k, v in output.items()}, camera, kpts
        )
        depth_dict[int(imid)] = depth

    return depth_dict
