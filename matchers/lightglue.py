from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace
from kornia.core import Module, ModuleList, Tensor
from kornia.core.check import KORNIA_CHECK
from kornia.feature.lightglue import LearnableFourierPositionalEncoding, TransformerLayer, MatchAssignment, TokenConfidence

from data import FilePath, resolve_model_path
from matchers.base import LocalFeatureMatcher
from matchers.config import LightGlueConfig
from postprocesses.panet import PANetRefiner
from preprocesses.region import OverlapRegionCropper
from storage import LocalFeatureStorage, MatchingStorage


def read_image(path: str) -> np.ndarray:
    return cv2.imread(str(path))


class _LightGlue(kornia.feature.LightGlue):
    def __init__(self,
                 state_dict: dict,
                 features: str = "superpoint", **conf_) -> None:  # type: ignore
        Module.__init__(self)
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf_})
        if features is not None:
            KORNIA_CHECK(features in list(self.features.keys()), "Features keys are wrong")
            for k, v in self.features[features].items():
                setattr(conf, k, v)
        KORNIA_CHECK(not (self.conf.add_scale_ori and self.conf.add_laf))  # we use either scale ori, or LAF

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()  # type: ignore

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * conf.add_scale_ori + 4 * conf.add_laf,
            head_dim,
            head_dim,
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim
        self.transformers = ModuleList([TransformerLayer(d, h, conf.flash) for _ in range(n)])
        self.log_assignment = ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = ModuleList([TokenConfidence(d) for _ in range(n - 1)])
        self.register_buffer(
            "confidence_thresholds",
            Tensor([self.confidence_threshold(i) for i in range(self.conf.n_layers)]),
        )
        # rename old state dict entries
        for i in range(self.conf.n_layers):
            pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
        incompatible_keys = self.load_state_dict(state_dict, strict=False)
        print(f"Loaded LightGlue model: {incompatible_keys}")
        # static lengths LightGlue is compiled for (only used with torch.compile)
        self.static_lengths = None


class _LightGlueMatcher(kornia.feature.LightGlueMatcher):
    def __init__(self,
                 weight_path: str | Path,
                 feature_name: str = "disk",
                 params: Optional[dict] = None) -> None:  # type: ignore
        params = params or {}
        feature_name_: str = feature_name.lower()
        kornia.feature.GeometryAwareDescriptorMatcher.__init__(self, feature_name_)
        self.feature_name = feature_name_
        self.params = params
        state_dict = torch.load(weight_path, map_location='cpu')
        self.matcher = _LightGlue(state_dict, self.feature_name, **params)
        print(f'[LightGlueMatcher] Loaded: {weight_path}')


class LightGlueMatcher(LocalFeatureMatcher):
    def __init__(self, conf: LightGlueConfig,
                 refiner: Optional[PANetRefiner] = None,
                 device: Optional[torch.device] = None):
        super().__init__(refiner=refiner)
        self.conf = conf
        self.device = device
        matcher = _LightGlueMatcher(
            resolve_model_path(conf.weight_path),
            feature_name=conf.feature_name,
            params=conf.get_params()
        ).eval().to(self.device)
        self.matcher = matcher

    @property
    def min_matches(self) -> Optional[int]:
        return self.conf.min_matches

    @property
    def use_overlap_region_cropper(self) -> bool:
        return self.conf.use_overlap_region_cropper

    @torch.inference_mode()
    def __call__(
        self,
        path1: FilePath,
        path2: FilePath,
        shape1: Tuple[int, int],    # (H, W)
        shape2: Tuple[int, int],    # (H, W)
        feature_storage: LocalFeatureStorage,
        matching_storage: Optional[MatchingStorage] = None,
        cropper: Optional[OverlapRegionCropper] = None,
        image_reader: Optional[Callable] = None
    ) -> np.ndarray:
        lafs1, kpts1, scores1, descs1 = feature_storage.get(path1)
        lafs2, kpts2, scores2, descs2 = feature_storage.get(path2)

        inputs = build_lightglue_input_data(
            kpts1, kpts2,
            descs1, descs2,
            shape1, shape2,
            scores1, scores2,
            device=self.device
        )

        dists, idxs = self.matcher(
            inputs['descriptors0'],
            inputs['descriptors1'],
            inputs['lafs0'],
            inputs['lafs1']
        )
        idxs = idxs.cpu().numpy()

        if self.use_overlap_region_cropper and cropper:
            idxs = self.filter_matches_out_of_overlap_region(
                idxs, kpts1, kpts2, cropper
            )

        if matching_storage:
            if self.min_matches is None or len(idxs) >= self.min_matches:
                matching_storage.add(path1, path2, idxs)
        return idxs

    def match(
        self,
        descs1: np.ndarray,
        descs2: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


def build_lightglue_input_data(
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    descs1: np.ndarray,
    descs2: np.ndarray,
    shape1: Tuple[int, int],
    shape2: Tuple[int, int],
    scores1: Optional[np.ndarray] = None,
    scores2: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None
) -> dict:
    assert scores1 is not None and scores2 is not None
    h0, w0, *_ = shape1
    h1, w1, *_ = shape2

    # Shape(N, dim)
    descriptors0 = torch.from_numpy(descs1).float().to(device, non_blocking=True)
    descriptors1 = torch.from_numpy(descs2).float().to(device, non_blocking=True)

    # Shape(N,)
    responses0 = torch.from_numpy(scores1).float().to(device, non_blocking=True)
    responses1 = torch.from_numpy(scores2).float().to(device, non_blocking=True)

    # Shape(1, N, 2)
    keypoints0 = torch.from_numpy(kpts1).float().unsqueeze(0).to(device, non_blocking=True)
    keypoints1 = torch.from_numpy(kpts2).float().unsqueeze(0).to(device, non_blocking=True)

    #lafs0 = torch.cat([
    #    torch.eye(2, device=device, dtype=keypoints0.dtype).unsqueeze(0).unsqueeze(1).expand(
    #        keypoints0.size(0), keypoints0.size(1), -1, -1
    #    ),
    #    keypoints0.unsqueeze(-1),
    #], dim=-1)
    #lafs1 = torch.cat([
    #    torch.eye(2, device=device, dtype=keypoints1.dtype).unsqueeze(0).unsqueeze(1).expand(
    #        keypoints1.size(0), keypoints1.size(1), -1, -1
    #    ),
    #    keypoints1.unsqueeze(-1),
    #], dim=-1)
    lafs0 = kornia.feature.laf_from_center_scale_ori(keypoints0)
    lafs1 = kornia.feature.laf_from_center_scale_ori(keypoints1)

    data = {
        'h0': h0,
        'w0': w0,
        'h1': h1,
        'w1': w1,
        'lafs0': lafs0,
        'lafs1': lafs1,
        'descriptors0': descriptors0,
        'descriptors1': descriptors1,
        'responses0': responses0,
        'responses1': responses1
    }
    return data