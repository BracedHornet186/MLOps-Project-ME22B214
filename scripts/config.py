from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import yaml
from pydantic import BaseModel

from pipelines.config import (
    DetectorBasedPipelineConfig,
    DetectorFreePipelineConfig,
    DUSt3RPipelineConfig,
    IMC2024PipelineConfig,
    IMC2025MASt3RPipelineConfig,
    IMC2025PipelineConfig,
    LocalizerPipelineConfig,
    PreMatchingEnsemblePipelineConfig,
    SimpleEnsemblePipelineConfig,
)

FilePath = Union[str, Path]

if TYPE_CHECKING:
    import torch


class DistConfig(BaseModel):
    gpu: int
    rank: int
    world_size: int

    @classmethod
    def single(cls) -> DistConfig:
        return DistConfig(gpu=0, rank=0, world_size=1)

    @property
    def device(self) -> "torch.device":
        import torch

        return torch.device(f"cuda:{self.gpu}")

    def ddp(self) -> bool:
        return self.world_size > 1

    def is_master(self) -> bool:
        return self.rank == 0

    def is_slave(self) -> bool:
        return self.rank > 0


class PipelineConfig(BaseModel):
    type: Literal[
        "detector_based",
        "detector_free",
        "pre_matching_ensemble",
        "simple_ensemble",
        "imc2024",
        "dust3r",
        "localizer",
        "kernel_debug",
        "imc2025",
        "imc2025_mast3r",
    ]

    detector_based_pipeline: Optional[DetectorBasedPipelineConfig] = None
    detector_free_pipeline: Optional[DetectorFreePipelineConfig] = None
    pre_matching_ensemble_pipeline: Optional[PreMatchingEnsemblePipelineConfig] = None
    simple_ensemble_pipeline: Optional[SimpleEnsemblePipelineConfig] = None
    imc2024_pipeline: Optional[IMC2024PipelineConfig] = None
    dust3r_pipeline: Optional[DUSt3RPipelineConfig] = None
    localizer_pipeline: Optional[LocalizerPipelineConfig] = None
    imc2025_pipeline: Optional[IMC2025PipelineConfig] = None
    imc2025_mast3r_pipeline: Optional[IMC2025MASt3RPipelineConfig] = None

    pipeline_id: Optional[str] = None

    @classmethod
    def load_config(cls, path: FilePath) -> PipelineConfig:
        import yaml

        with open(path) as fp:
            conf = PipelineConfig.parse_obj(yaml.safe_load(fp))

        conf.pipeline_id = Path(path).stem
        return conf

    def get_core_config(
        self,
    ) -> (
        DetectorBasedPipelineConfig
        | DetectorFreePipelineConfig
        | PreMatchingEnsemblePipelineConfig
        | SimpleEnsemblePipelineConfig
        | IMC2024PipelineConfig
        | DUSt3RPipelineConfig
        | LocalizerPipelineConfig
        | IMC2025PipelineConfig
        | IMC2025MASt3RPipelineConfig
    ):
        if self.type == "detector_based":
            assert self.detector_based_pipeline
            return self.detector_based_pipeline
        elif self.type == "detector_free":
            assert self.detector_free_pipeline
            return self.detector_free_pipeline
        elif self.type == "pre_matching_ensemble":
            assert self.pre_matching_ensemble_pipeline
            return self.pre_matching_ensemble_pipeline
        elif self.type == "simple_ensemble":
            assert self.simple_ensemble_pipeline
            return self.simple_ensemble_pipeline
        elif self.type == "imc2024":
            assert self.imc2024_pipeline
            return self.imc2024_pipeline
        elif self.type == "dust3r":
            assert self.dust3r_pipeline
            return self.dust3r_pipeline
        elif self.type == "localizer":
            assert self.localizer_pipeline
            return self.localizer_pipeline
        elif self.type == "kernel_debug":
            assert self.imc2024_pipeline
            return self.imc2024_pipeline
        elif self.type == "imc2025":
            assert self.imc2025_pipeline
            return self.imc2025_pipeline
        elif self.type == "imc2025_mast3r":
            assert self.imc2025_mast3r_pipeline
            return self.imc2025_mast3r_pipeline
        raise ValueError


class SubmissionConfig(BaseModel):
    pipeline: PipelineConfig

    target_data_type: Literal[
        "submission",
        "submission-fast-commit",
        "debug",
        "imc2025test",
        "imc2025train",
        "imc2024test",
        "imc2024train",
        "imc2023test",
        "imc2023train",
    ] = "submission"

    # For IMC25
    datasets_to_ignore: Optional[list[str]] = None
    datasets_to_use: Optional[list[str]] = None

    # For IMC24
    scenes_to_ignore: Optional[list[str]] = None
    scenes_to_use: Optional[list[str]] = None

    dist_output_dir: str = ".dist_output"

    @classmethod
    def load_config(cls, path: FilePath) -> SubmissionConfig:
        with open(path) as fp:
            return SubmissionConfig.model_validate(yaml.safe_load(fp))

    @classmethod
    def load_config_from_string(cls, content: str) -> SubmissionConfig:
        return SubmissionConfig.model_validate(yaml.safe_load(content))

    @classmethod
    def load_config_from_pipeline_config_string(
        cls, content: str, **kwargs
    ) -> SubmissionConfig:
        pipeline = PipelineConfig.model_validate(yaml.safe_load(content))
        return SubmissionConfig(pipeline=pipeline, **kwargs)

    @property
    def dist_output_dir_path(self) -> Path:
        return Path(self.dist_output_dir)


def load_pipeline_config(
    path: FilePath,
    *,
    target_data_type: Literal[
        "submission",
        "submission-fast-commit",
        "debug",
        "imc2025test",
        "imc2025train",
        "imc2024test",
        "imc2024train",
        "imc2023test",
        "imc2023train",
    ] = "imc2025train",
) -> SubmissionConfig:
    """Load a pipeline YAML into the same SubmissionConfig flow used by kernel scripts."""
    return SubmissionConfig(
        pipeline=PipelineConfig.load_config(path),
        target_data_type=target_data_type,
    )
