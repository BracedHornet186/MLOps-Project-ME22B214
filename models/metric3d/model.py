"""Adapted from https://github.com/YvanYin/Metric3D"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from mmengine import Config
from mono.model.monodepth_model import DepthModel, get_configured_monodepth_model
from mono.utils.do_test import resize_for_input


class Metric3DDepthModel:
    def __init__(
        self,
        weight_path: str | Path,
        config_path: str | Path,
        device: torch.device,
    ):
        model, cfg = load_metric3d_model(weight_path, config_path)
        self.model = model.eval().to(device)
        self.cfg = cfg
        self.device = device

    def predict(self, image: np.ndarray, intrinsics: np.ndarray):
        ori_shape = [image.shape[0], image.shape[1]]
        rgb_input, cam_models_stacks, pad_info, label_scale_factor = (
            transform_test_data_scalecano_fixed(
                image, intrinsics, self.cfg.data_basic, self.device
            )
        )
        rgb_input = rgb_input[None]
        normalize_scale = self.cfg.data_basic.depth_range[1]
        data = dict(
            input=rgb_input,
            cam_model=None,  # default method inputs cam model but doesn not use it
        )

        _, _, output = self.model.inference(data)
        pred_depth_canon, pred_normal_canon_, confidence_canon = [
            output[key].squeeze(0)
            for key in ["prediction", "prediction_normal", "confidence"]
        ]

        pred_normal_canon = pred_normal_canon_[:-1]
        normal_confidence_canon = pred_normal_canon_[-1, None]
        valid_canon = (pred_depth_canon < 200).float()

        pred_depth, valid, pred_normal, confidence, normal_confidence = [
            slice_and_interpolate(tensor, pad_info, ori_shape)
            for tensor in [
                pred_depth_canon,
                valid_canon,
                pred_normal_canon,
                confidence_canon,
                normal_confidence_canon,
            ]
        ]

        pred_depth = pred_depth * normalize_scale / label_scale_factor
        confidence = torch.clamp(confidence, 0, 1)
        error = pred_depth * (1 - confidence)
        normals = omni_to_bni(pred_normal.permute(1, 2, 0))
        depth_variance = error**2
        outdict = dict(
            depth=pred_depth,
            depth_variance=depth_variance,
            normals=normals,
            normals_confidence=normal_confidence,
            valid=valid == 1,
        )
        return outdict


def load_metric3d_model(
    weight_path: str | Path,
    config_path: str | Path,
) -> tuple[DepthModel, Config]:
    cfg = Config.fromfile(config_path)
    model = get_configured_monodepth_model(cfg)
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict["model_state_dict"], strict=False)
    assert isinstance(model, DepthModel)
    return model, cfg


def slice_and_interpolate(
    tensor: torch.Tensor, pad_info, ori_shape: list[int], mode: str = "bilinear"
) -> torch.Tensor:
    sliced_tensor = tensor[
        :,
        pad_info[0] : tensor.shape[1] - pad_info[1],
        pad_info[2] : tensor.shape[2] - pad_info[3],
    ]
    return torch.nn.functional.interpolate(
        sliced_tensor[None], ori_shape, mode=mode
    ).squeeze(0)


def transform_test_data_scalecano_fixed(
    rgb: np.ndarray,
    intrinsic: np.ndarray,
    data_basic,
    device: torch.device,
):
    """
    Pre-process the input for forwarding. Employ `label scale canonical transformation.'
        Args:
            rgb: input rgb image. [H, W, 3]
            intrinsic: camera intrinsic parameter, [fx, fy, u0, v0]
            data_basic: predefined canonical space in configs.
    """
    canonical_space = data_basic["canonical_space"]
    forward_size = data_basic.crop_size
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

    # BGR to RGB
    # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    ori_h, ori_w, _ = rgb.shape
    ori_focal = (intrinsic[0] + intrinsic[1]) / 2
    canonical_focal = canonical_space["focal_length"]

    cano_label_scale_ratio = canonical_focal / ori_focal

    canonical_intrinsic = [
        intrinsic[0] * cano_label_scale_ratio,
        intrinsic[1] * cano_label_scale_ratio,
        intrinsic[2],
        intrinsic[3],
    ]

    # resize
    rgb, cam_model, pad, resize_label_scale_ratio = resize_for_input(
        rgb, forward_size, canonical_intrinsic, [ori_h, ori_w], 1.0
    )

    # label scale factor
    label_scale_factor = cano_label_scale_ratio * resize_label_scale_ratio

    rgb: torch.Tensor = torch.from_numpy(rgb.transpose((2, 0, 1))).float()  # type: ignore
    rgb = torch.div((rgb - mean), std)
    rgb = rgb.to(device)

    cam_model = torch.from_numpy(cam_model.transpose((2, 0, 1))).float()
    cam_model = cam_model[None, :, :, :].to(device)
    cam_model_stacks = [
        torch.nn.functional.interpolate(
            cam_model,
            size=(cam_model.shape[2] // i, cam_model.shape[3] // i),
            mode="bilinear",
            align_corners=False,
        )
        for i in [2, 4, 8, 16, 32]
    ]
    return rgb, cam_model_stacks, pad, label_scale_factor


def omni_to_bni(normals: torch.Tensor) -> torch.Tensor:
    normals[..., 1:] = -normals[..., 1:]
    return normals
