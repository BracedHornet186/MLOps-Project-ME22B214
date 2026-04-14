from typing import Optional

import numpy as np
import torch
from DeDoDe.utils import dual_softmax_matcher

from matchers.base import LocalFeatureMatcher
from matchers.config import DualSoftmaxMatcherConfig
from postprocesses.panet import PANetRefiner


class DualSoftmaxMatcher(LocalFeatureMatcher):
    def __init__(
        self,
        conf: DualSoftmaxMatcherConfig,
        refiner: Optional[PANetRefiner] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(refiner=refiner)
        self.conf = conf
        self.device = device

    @property
    def min_matches(self) -> Optional[int]:
        return self.conf.min_matches

    @property
    def use_overlap_region_cropper(self) -> bool:
        return self.conf.use_overlap_region_cropper

    @torch.inference_mode()
    def match(self, descs1: np.ndarray, descs2: np.ndarray) -> np.ndarray:
        x1 = torch.from_numpy(descs1).float().to(self.device, non_blocking=True)
        x2 = torch.from_numpy(descs2).float().to(self.device, non_blocking=True)

        P = dual_softmax_matcher(
            x1, # type: ignore
            x2, # type: ignore
            inv_temperature=self.conf.inv_temp, # type: ignore
            normalize=self.conf.normalize,
        )
        inds = torch.nonzero(
            (P == P.max(dim=-1, keepdim=True).values)
            * (P == P.max(dim=-2, keepdim=True).values)
            * (P > self.conf.threshold)
        )   # Shape(N, 3)
        idxs = inds[:, 1:]

        assert isinstance(idxs, torch.Tensor)
        idxs = idxs.detach().cpu().numpy()
        return idxs
