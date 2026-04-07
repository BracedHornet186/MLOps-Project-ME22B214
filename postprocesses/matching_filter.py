from typing import Optional

import torch

from postprocesses.base import MatchingFilter
from postprocesses.config import MatchingFilterConfig
from postprocesses.feature_track import FeatureTrackMatchingFilter
from postprocesses.ncmnet import NCMNetFilter


def create_matching_filter(
    conf: MatchingFilterConfig,
    device: Optional[torch.device] = None,
) -> MatchingFilter:
    if conf.type == "ncmnet":
        assert conf.ncmnet
        return NCMNetFilter(conf.ncmnet, device=device)
    elif conf.type == "feature_track":
        return FeatureTrackMatchingFilter()
    raise ValueError(conf.type)
