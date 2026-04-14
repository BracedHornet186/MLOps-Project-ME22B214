from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from kornia.feature.loftr.loftr import LoFTR
from lightglue import LightGlue
from lightglue.utils import rbd

from features.lightglue.superpoint import _SuperPoint


class MINIMASuperPointLightGlueMatcher(torch.nn.Module):
    def __init__(
        self,
        sp_weight_path: str | Path,
        lg_weight_path: str | Path,
        sp_conf: dict,
        lg_conf: dict,
        device: torch.device,
    ):
        super().__init__()
        self.extractor = (
            _SuperPoint(str(sp_weight_path), **sp_conf).eval().to(device)
        )  # load the feature extractor
        self.matcher = (
            LightGlue(features="superpoint", **lg_conf).eval().to(device)
        )  # load the matcher
        n_layers = lg_conf["n_layers"]
        # print(f"n_layers: {n_layers}")
        # rename old state dict entries
        state_dict = torch.load(str(lg_weight_path), map_location=device)
        for i in range(n_layers):
            pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
        self.matcher.load_state_dict(state_dict, strict=False)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # batch = {'image0': image0, 'image1': image1}
        image0 = batch["image0"]
        image1 = batch["image1"]
        feats0 = self.extractor.extract(
            image0
        )  # auto-resize the image, disable with resize=None
        feats1 = self.extractor.extract(image1)

        # match the features
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension
        matches = matches01["matches"]  # indices with shape (K,2)
        points0 = feats0["keypoints"][
            matches[..., 0]
        ]  # coordinates in image #0, shape (K,2)
        points1 = feats1["keypoints"][
            matches[..., 1]
        ]  # coordinates in image #1, shape (K,2)
        matching_scores0 = matches01["matching_scores0"]
        matching_scores = matching_scores0[matches[..., 0]]

        return {
            "matching_scores": matching_scores,
            "keypoints0": points0,
            "keypoints1": points1,
        }


def create_minima_lightglue(
    sp_weight_path: str | Path,
    lg_weight_path: str | Path,
    device: torch.device,
) -> MINIMASuperPointLightGlueMatcher:
    sp_conf = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": 2048,
        "detection_threshold": 0.0005,
        "remove_borders": 4,
    }
    lg_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": True,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": -1,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }
    return MINIMASuperPointLightGlueMatcher(
        sp_weight_path,
        lg_weight_path,
        sp_conf,
        lg_conf,
        device,
    )


def create_minima_loftr(
    weight_path: str | Path,
    device: torch.device,
) -> LoFTR:
    model = LoFTR(pretrained=None)
    weight = torch.load(str(weight_path), map_location="cpu")
    model.load_state_dict(weight["state_dict"])
    model = model.eval().to(device)
    return model
