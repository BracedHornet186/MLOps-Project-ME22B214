from __future__ import annotations

from typing import Optional

import torch

from config import PipelineConfig
from distributed import DistConfig
from pipelines.base import Pipeline
from pipelines.imc2025_mast3r_pipeline import IMC2025MASt3RPipeline
from pipelines.imc2025_pipeline import IMC2025Pipeline


def create_pipeline(
    conf: PipelineConfig,
    dist_conf: Optional[DistConfig] = None,
    device: Optional[torch.device] = None,
) -> Pipeline:
    if conf.type == "imc2025":
        assert conf.imc2025_pipeline
        pipeline = IMC2025Pipeline(
            conf.imc2025_pipeline,
            dist_conf=dist_conf,
            device=device,
        )
    elif conf.type == "imc2025_mast3r":
        assert conf.imc2025_mast3r_pipeline
        pipeline = IMC2025MASt3RPipeline(
            conf.imc2025_mast3r_pipeline,
            dist_conf=dist_conf,
            device=device,
        )
    else:
        raise ValueError(conf.type)

    pipeline.set_id(conf.pipeline_id)
    return pipeline
