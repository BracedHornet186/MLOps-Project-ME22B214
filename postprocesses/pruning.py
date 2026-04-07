from __future__ import annotations

from typing import Optional

import torch

from pipelines.config import TwoViewGeometryPruningConfig
from postprocesses.base import TwoViewGeometryPruner
from postprocesses.doppelgangers import DoppelGangersTwoViewGeometryPruner
from postprocesses.vggt_verifier import VGGTTwoViewGeometryPruner


def create_pruner(
    conf: TwoViewGeometryPruningConfig, device: Optional[torch.device] = None
) -> TwoViewGeometryPruner:
    if conf.type == "doppelgangers":
        assert conf.doppelgangers
        pruner = DoppelGangersTwoViewGeometryPruner(conf.doppelgangers, device=device)
    elif conf.type == "vggt":
        assert conf.vggt
        assert device is not None
        pruner = VGGTTwoViewGeometryPruner(conf.vggt, device=device)
    else:
        raise ValueError(conf.type)
    return pruner
