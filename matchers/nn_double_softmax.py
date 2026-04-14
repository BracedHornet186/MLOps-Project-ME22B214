from typing import Optional
import kornia
import numpy as np
import torch

from matchers.base import LocalFeatureMatcher
from matchers.config import NNDoubleSoftmaxConfig
from postprocesses.panet import PANetRefiner


class NNDoubleSoftmaxMatcher(LocalFeatureMatcher):
    def __init__(self, conf: NNDoubleSoftmaxConfig,
                 refiner: Optional[PANetRefiner] = None,
                 device: Optional[torch.device] = None):
        super().__init__(refiner=refiner)
        self.conf = conf
        self.device = device

        matcher_kwargs = {}
        if conf.th is not None:
            matcher_kwargs['th'] = conf.th

    @property
    def min_matches(self) -> Optional[int]:
        return self.conf.min_matches

    @property
    def use_overlap_region_cropper(self) -> bool:
        return self.conf.use_overlap_region_cropper
    
    @torch.inference_mode()
    def match(self,
              descs1: np.ndarray,
              descs2: np.ndarray) -> np.ndarray:
        x1 = torch.from_numpy(descs1).float().to(self.device, non_blocking=True)
        x2 = torch.from_numpy(descs2).float().to(self.device, non_blocking=True)
        sim = x1 @ x2.T / self.conf.temperature
        prob = torch.softmax(sim, dim=0) * torch.softmax(sim, dim=1)
        dm = 1.0 - prob
        if self.conf.match_mode == 'nn':
            dists, idxs = kornia.feature.match_nn(x1, x2, dm=dm)
        elif self.conf.match_mode == 'mnn':
            dists, idxs = kornia.feature.match_mnn(x1, x2, dm=dm)
        elif self.conf.match_mode == 'snn':
            dists, idxs = kornia.feature.match_snn(x1, x2, dm=dm)
        elif self.conf.match_mode == 'smnn':
            dists, idxs = kornia.feature.match_smnn(x1, x2, dm=dm)
        else:
            raise ValueError(self.conf.match_mode)

        assert isinstance(idxs, torch.Tensor)
        idxs = idxs.detach().cpu().numpy()
        return idxs
