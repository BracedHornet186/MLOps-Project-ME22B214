from __future__ import annotations

from typing import Optional

import torch

from config import PipelineConfig
from distributed import DistConfig
from pipelines.base import Pipeline
from pipelines.imc2023.detector_based_pipeline import DetectorBasedPipeline
from pipelines.imc2023.detector_free_pipeline import DetectorFreePipeline
from pipelines.imc2023.prematching_ensemble_pipeline import PreMatchingEnsemblePipeline
from pipelines.imc2023.simple_ensemble_pipeline import SimpleEnsemblePipeline
from pipelines.imc2024.imc2024_pipeline import IMC2024Pipeline
from pipelines.imc2024.kernel_debug_pipeline import KernelDebugPipeline
from pipelines.imc2025_mast3r_pipeline import IMC2025MASt3RPipeline
from pipelines.imc2025_pipeline import IMC2025Pipeline


def create_pipeline(
    conf: PipelineConfig,
    dist_conf: Optional[DistConfig] = None,
    device: Optional[torch.device] = None,
) -> Pipeline:
    if conf.type == "detector_based":
        assert conf.detector_based_pipeline
        pipeline = DetectorBasedPipeline(
            conf.detector_based_pipeline, dist_conf=dist_conf, device=device
        )
    elif conf.type == "detector_free":
        assert conf.detector_free_pipeline
        pipeline = DetectorFreePipeline(
            conf.detector_free_pipeline, dist_conf=dist_conf, device=device
        )
    elif conf.type == "pre_matching_ensemble":
        assert conf.pre_matching_ensemble_pipeline
        pipeline = PreMatchingEnsemblePipeline(
            conf.pre_matching_ensemble_pipeline, dist_conf=dist_conf, device=device
        )
    elif conf.type == "simple_ensemble":
        assert conf.simple_ensemble_pipeline
        pipeline = SimpleEnsemblePipeline(
            conf.simple_ensemble_pipeline, dist_conf=dist_conf, device=device
        )
    elif conf.type == "imc2024":
        assert conf.imc2024_pipeline
        pipeline = IMC2024Pipeline(
            conf.imc2024_pipeline, dist_conf=dist_conf, device=device
        )
    elif conf.type == "dust3r":
        assert conf.dust3r_pipeline
        from pipelines.imc2024.dust3r_pipeline import DUSt3RPipeline

        pipeline = DUSt3RPipeline(
            conf.dust3r_pipeline, dist_conf=dist_conf, device=device
        )
    elif conf.type == "localizer":
        assert conf.localizer_pipeline
        from pipelines.imc2024.localizer_pipeline import LocalizerPipeline

        pipeline = LocalizerPipeline(
            conf.localizer_pipeline, dist_conf=dist_conf, device=device
        )
    elif conf.type == "kernel_debug":
        assert conf.imc2024_pipeline
        pipeline = KernelDebugPipeline(
            conf.imc2024_pipeline, dist_conf=dist_conf, device=device
        )
    elif conf.type == "imc2025":
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
