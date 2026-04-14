from typing import Optional
import kornia
import numpy as np
import torch

from matchers.base import LocalFeatureMatcher
from matchers.config import NNMatcherConfig
from postprocesses.panet import PANetRefiner


class NNMatcher(LocalFeatureMatcher):
    def __init__(self, conf: NNMatcherConfig,
                 refiner: Optional[PANetRefiner] = None,
                 device: Optional[torch.device] = None):
        super().__init__(refiner=refiner)
        self.conf = conf
        self.device = device or torch.device('cpu')

        matcher_kwargs = {}
        if conf.th is not None:
            matcher_kwargs['th'] = conf.th

        self.matcher = kornia.feature.DescriptorMatcher(
            match_mode=conf.match_mode, **matcher_kwargs
        ).to(device)
        print(f'[NNMatcher] Use device = {self.device}')
    
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
        dists, idxs = self.matcher(x1, x2)
        assert isinstance(idxs, torch.Tensor)
        idxs = idxs.detach().cpu().numpy()
        return idxs
