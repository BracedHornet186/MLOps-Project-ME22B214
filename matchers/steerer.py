from typing import Callable, Optional

import numpy as np
import torch
from DeDoDe.utils import dual_softmax_matcher

from scripts.data import FilePath, resolve_model_path
from matchers.base import LocalFeatureMatcher
from matchers.config import SteererConfig
from models.rotation_steerers.matchers.max_matches import (
    ContinuousMaxMatchesMatcher,
    ContinuousSubsetMatcher,
    MaxMatchesMatcher,
    SubsetMatcher,
)
from models.rotation_steerers.matchers.max_similarity import (
    ContinuousMaxSimilarityMatcher,
    MaxSimilarityMatcher,
)
from models.rotation_steerers.steerers import ContinuousSteerer, DiscreteSteerer
from postprocesses.panet import PANetRefiner
from preprocesses.region import OverlapRegionCropper
from scripts.storage import LocalFeatureStorage, MatchingStorage


class SteererMatcher(LocalFeatureMatcher):
    def __init__(
        self,
        conf: SteererConfig,
        refiner: Optional[PANetRefiner] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(refiner=refiner)
        self.conf = conf
        self.device = device

        weight_path = str(resolve_model_path(conf.weight_path))
        if conf.matcher_type in (
            "MaxMatchesMatcher",
            "SubsetMatcher",
            "MaxSimilarityMatcher",
        ):
            steerer_generator = torch.load(weight_path, map_location=self.device)
            steerer = DiscreteSteerer(steerer_generator)
            if conf.so2:
                steerer.generator = torch.matrix_exp(0.25 * np.pi * steerer.generator)

            if conf.matcher_type == "MaxMatchesMatcher":
                assert conf.steerer_order
                matcher = MaxMatchesMatcher(
                    steerer=steerer, steerer_order=conf.steerer_order
                )
            elif conf.matcher_type == "MaxSimilarityMatcher":
                assert conf.steerer_order
                matcher = MaxSimilarityMatcher(
                    steerer=steerer, steerer_order=conf.steerer_order
                )
            elif conf.matcher_type == "SubsetMatcher":
                assert conf.steerer_order
                matcher = SubsetMatcher(
                    steerer=steerer, steerer_order=conf.steerer_order
                )
                raise NotImplementedError
            else:
                raise ValueError(conf.matcher_type)
        elif conf.matcher_type in ("ContinuousMaxMatchesMatcher",):
            steerer_generator = torch.load(weight_path, map_location=self.device)
            steerer = ContinuousSteerer(steerer_generator)
            if conf.matcher_type == "ContinuousMaxMatchesMatcher":
                matcher = ContinuousMaxMatchesMatcher(
                    steerer=steerer, angles=conf.angles
                )
                raise NotImplementedError
            else:
                raise ValueError(conf.matcher_type)
        else:
            raise ValueError(conf.matcher_type)

        self.matcher = matcher.eval().to(self.device)

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
        shape1: tuple[int, int],  # (H, W)
        shape2: tuple[int, int],  # (H, W)
        feature_storage: LocalFeatureStorage,
        matching_storage: Optional[MatchingStorage] = None,
        cropper: Optional[OverlapRegionCropper] = None,
        image_reader: Optional[Callable] = None,
    ) -> np.ndarray:
        lafs1, kpts1, scores1, descs1 = feature_storage.get(path1)
        lafs2, kpts2, scores2, descs2 = feature_storage.get(path2)
        h1, w1 = shape1
        h2, w2 = shape2

        _kpts1 = torch.from_numpy(kpts1).to(self.device, non_blocking=True)
        _kpts2 = torch.from_numpy(kpts2).to(self.device, non_blocking=True)
        _scores1 = torch.from_numpy(scores1).to(self.device, non_blocking=True)
        _scores2 = torch.from_numpy(scores2).to(self.device, non_blocking=True)
        _descs1 = torch.from_numpy(descs1).to(self.device, non_blocking=True)
        _descs2 = torch.from_numpy(descs2).to(self.device, non_blocking=True)
        _, _, idxs = self.matcher.match(
            _kpts1[None],
            _descs1[None],
            _kpts2[None],
            _descs2[None],
            P_A=_scores1[None],
            P_B=_scores2[None],
            normalize=self.conf.normalize,
            inv_temp=self.conf.inv_temp,  # type: ignore
            threshold=self.conf.threshold,
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

    @torch.inference_mode()
    def match(self, descs1: np.ndarray, descs2: np.ndarray) -> np.ndarray:
        raise NotImplementedError
