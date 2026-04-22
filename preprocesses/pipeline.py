"""
preprocesses/pipeline.py
────────────────────────
Central orchestrator for all image-level preprocessing steps.
Each step is independently configurable via PreprocessingConfig and
writes its outputs to Scene so downstream matching stages read them
transparently.

Preprocessing order (all steps optional, controlled by config):
  1. Orientation normalization  (CheckOrientationHandler)
  2. Deblurring                 (FFTformerDeblurHandler)
  3. Segmentation               (GroundedSAMSegmentator)
  4. Depth estimation           (HFDepthEstimationModel)
"""

from __future__ import annotations

import os
# Configure PyTorch to use expandable segments for improved high-res image memory fragmentation handling
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlflow
import torch
import tqdm

from pipelines.scene import Scene
from preprocesses.config import (
    DeblurringConfig,
    DepthEstimationConfig,
    OrientationNormalizationConfig,
    SegmentationConfig,
)

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessingConfig:
    """
    Master config that toggles each preprocessing step on/off.
    Mirrors the YAML structure at conf/preprocess.yaml.
    """
    orientation: Optional[OrientationNormalizationConfig] = None
    deblurring: Optional[DeblurringConfig] = None
    segmentation: Optional[SegmentationConfig] = None
    depth_estimation: Optional[DepthEstimationConfig] = None

    # Logging / tracking
    log_to_mlflow: bool = True
    mlflow_run_name: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PreprocessingConfig":
        import yaml
        from preprocesses.config import (
            OrientationNormalizationConfig, DeblurringConfig,
            SegmentationConfig, DepthEstimationConfig,
        )
        with open(path) as f:
            raw = yaml.safe_load(f)
        kwargs: dict = {}
        if raw.get("orientation"):
            kwargs["orientation"] = OrientationNormalizationConfig(**raw["orientation"])
        if raw.get("deblurring"):
            kwargs["deblurring"] = DeblurringConfig(**raw["deblurring"])
        if raw.get("segmentation"):
            kwargs["segmentation"] = SegmentationConfig(**raw["segmentation"])
        if raw.get("depth_estimation"):
            kwargs["depth_estimation"] = DepthEstimationConfig(**raw["depth_estimation"])
        kwargs["log_to_mlflow"] = raw.get("log_to_mlflow", True)
        kwargs["mlflow_run_name"] = raw.get("mlflow_run_name")
        return cls(**kwargs)

    def active_steps(self) -> list[str]:
        steps = []
        if self.orientation:
            steps.append("orientation")
        if self.deblurring:
            steps.append("deblurring")
        if self.segmentation:
            steps.append("segmentation")
        if self.depth_estimation:
            steps.append("depth_estimation")
        return steps


# ── Per-step stats ─────────────────────────────────────────────────────────────

@dataclass
class StepStats:
    step_name: str
    images_processed: int = 0
    images_skipped: int = 0
    images_failed: int = 0
    elapsed_sec: float = 0.0

    def as_dict(self) -> dict:
        return {
            f"{self.step_name}_processed": self.images_processed,
            f"{self.step_name}_skipped": self.images_skipped,
            f"{self.step_name}_failed": self.images_failed,
            f"{self.step_name}_elapsed_sec": round(self.elapsed_sec, 2),
        }


# ── Main pipeline class ────────────────────────────────────────────────────────

class PreprocessingPipeline:
    """
    Runs all configured preprocessing steps on a Scene in order.
    Each step updates scene in-place (deblurred_images, depth_images,
    segmentation_mask_images, orientations dicts).
    """

    def __init__(
        self,
        conf: PreprocessingConfig,
        device: Optional[torch.device] = None,
    ):
        self.conf = conf
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            f"PreprocessingPipeline initialised | device={self.device} | "
            f"steps={conf.active_steps()}"
        )

    def run(
        self,
        scene: Scene,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> Scene:
        """
        Execute all enabled preprocessing steps.
        Returns the same Scene object with in-place updates.
        """
        all_stats: list[StepStats] = []
        run_params = {
            "n_images": len(scene.image_paths),
            "active_steps": str(self.conf.active_steps()),
            "device": str(self.device),
        }

        mlflow_ctx = (
            mlflow.start_run(
                run_name=self.conf.mlflow_run_name or f"preprocess_{scene.scene}",
                nested=True,
            )
            if self.conf.log_to_mlflow
            else _NullCtx()
        )

        with mlflow_ctx:
            if self.conf.log_to_mlflow:
                mlflow.log_params(run_params)

            # ── Step 1: Orientation ────────────────────────────────────────
            if self.conf.orientation:
                stats = self._run_orientation(scene, progress_bar)
                all_stats.append(stats)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ── Step 2: Deblurring ─────────────────────────────────────────
            if self.conf.deblurring:
                stats = self._run_deblurring(scene, progress_bar)
                all_stats.append(stats)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ── Step 3: Segmentation ───────────────────────────────────────
            if self.conf.segmentation:
                stats = self._run_segmentation(scene, progress_bar)
                all_stats.append(stats)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ── Step 4: Depth estimation ───────────────────────────────────
            if self.conf.depth_estimation:
                stats = self._run_depth_estimation(scene, progress_bar)
                all_stats.append(stats)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ── Log aggregated metrics ─────────────────────────────────────
            if self.conf.log_to_mlflow:
                metrics: dict = {}
                for s in all_stats:
                    metrics.update(s.as_dict())
                mlflow.log_metrics(metrics)

        return scene

    # ── Private step runners ───────────────────────────────────────────────────

    def _run_orientation(
        self,
        scene: Scene,
        progress_bar: Optional[tqdm.tqdm],
    ) -> StepStats:
        from preprocesses.orientation import compute_and_register_orientations
        stats = StepStats("orientation")
        t0 = time.perf_counter()
        try:
            logger.info(f"[{scene.scene}] Running orientation normalization...")
            compute_and_register_orientations(
                scene, self.conf.orientation, progress_bar=progress_bar
            )
            stats.images_processed = len(scene.image_paths)
            logger.info(
                f"[{scene.scene}] Orientation done: "
                f"{len(scene.orientations)} images updated"
            )
        except Exception as exc:
            logger.error(f"[{scene.scene}] Orientation step failed: {exc}")
            stats.images_failed = len(scene.image_paths)
        stats.elapsed_sec = time.perf_counter() - t0
        return stats

    def _run_deblurring(
        self,
        scene: Scene,
        progress_bar: Optional[tqdm.tqdm],
    ) -> StepStats:
        import cv2
        from preprocesses.deblur import run_deblurring
        stats = StepStats("deblurring")
        t0 = time.perf_counter()
        try:
            logger.info(
                f"[{scene.scene}] Running deblurring "
                f"(threshold={self.conf.deblurring.blurry_threshold})..."
            )
            n_before = len(scene.deblurred_images)
            run_deblurring(
                scene, self.conf.deblurring,
                device=self.device, progress_bar=progress_bar,
            )
            stats.images_processed = len(scene.deblurred_images) - n_before
            stats.images_skipped = (
                len(scene.image_paths) - stats.images_processed
            )
            logger.info(
                f"[{scene.scene}] Deblurring done: "
                f"{stats.images_processed} deblurred, "
                f"{stats.images_skipped} sharp (skipped)"
            )
        except Exception as exc:
            logger.error(f"[{scene.scene}] Deblurring step failed: {exc}")
            stats.images_failed = len(scene.image_paths)
        stats.elapsed_sec = time.perf_counter() - t0
        return stats

    def _run_segmentation(
        self,
        scene: Scene,
        progress_bar: Optional[tqdm.tqdm],
    ) -> StepStats:
        from preprocesses.segmentation import run_segmentation
        stats = StepStats("segmentation")
        t0 = time.perf_counter()
        try:
            skip = (
                self.conf.segmentation.skip_when_identical_camera_scene
                and scene.get_unique_resolution_num() == 1
            )
            if skip:
                logger.info(
                    f"[{scene.scene}] Segmentation skipped "
                    "(identical camera scene)"
                )
                stats.images_skipped = len(scene.image_paths)
            else:
                logger.info(f"[{scene.scene}] Running segmentation...")
                run_segmentation(
                    scene, self.conf.segmentation,
                    device=self.device, progress_bar=progress_bar,
                )
                stats.images_processed = len(scene.image_paths)
        except Exception as exc:
            logger.error(f"[{scene.scene}] Segmentation step failed: {exc}")
            stats.images_failed = len(scene.image_paths)
        stats.elapsed_sec = time.perf_counter() - t0
        return stats

    def _run_depth_estimation(
        self,
        scene: Scene,
        progress_bar: Optional[tqdm.tqdm],
    ) -> StepStats:
        from preprocesses.depth import run_depth_estimation
        stats = StepStats("depth_estimation")
        t0 = time.perf_counter()
        try:
            logger.info(f"[{scene.scene}] Running depth estimation...")
            run_depth_estimation(
                scene, self.conf.depth_estimation,
                device=self.device, progress_bar=progress_bar,
            )
            stats.images_processed = len(scene.image_paths)
            logger.info(
                f"[{scene.scene}] Depth estimation done: "
                f"{len(scene.depth_images)} depth maps cached"
            )
        except Exception as exc:
            logger.error(f"[{scene.scene}] Depth estimation step failed: {exc}")
            stats.images_failed = len(scene.image_paths)
        stats.elapsed_sec = time.perf_counter() - t0
        return stats


# ── Null context manager (when MLflow logging is disabled) ─────────────────────

class _NullCtx:
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
