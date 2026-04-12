"""
features/engineer.py
─────────────────────
Feature Engineering orchestrator for Stage 2.
Handles two feature families:

  A. Global descriptors  — compact image-level vectors used for
     shortlist generation (image retrieval).
     Models: DINOv2, ISC, MASt3R-SPoC, MASt3R-ASMK, SigLIP2

  B. Local features      — keypoint + descriptor pairs used for
     pairwise matching.
     Models: ALIKED+LightGlue, SuperPoint+LightGlue

All extracted features are:
  1. Saved to disk under data/processed/features/<scene>/
  2. Logged (counts, norms, timing) to MLflow
  3. Compared against EDA baselines for drift detection

Usage:
  from features.engineer import FeatureEngineer, FeatureEngineeringConfig
  cfg = FeatureEngineeringConfig.from_yaml("conf/features.yaml")
  eng = FeatureEngineer(cfg, device=torch.device("cuda"))
  result = eng.run(scene)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from pipelines.scene import Scene

logger = logging.getLogger(__name__)


# ── Sub-configs ────────────────────────────────────────────────────────────────

@dataclass
class GlobalDescriptorSpec:
    """Which global descriptor models to extract."""
    dinov2: bool = True
    isc: bool = True
    mast3r_spoc: bool = False      # requires MASt3R weights
    siglip2: bool = False          # requires SigLIP2 weights
    batch_size: int = 8


@dataclass
class LocalFeatureSpec:
    """Which local feature extractors to run."""
    aliked: bool = True
    superpoint: bool = True
    aliked_max_keypoints: int = 4096
    superpoint_max_keypoints: int = 4096
    aliked_resize: int = 1280
    superpoint_resize: int = 1600


@dataclass
class FeatureEngineeringConfig:
    global_descriptors: GlobalDescriptorSpec = field(
        default_factory=GlobalDescriptorSpec
    )
    local_features: LocalFeatureSpec = field(
        default_factory=LocalFeatureSpec
    )
    output_dir: str = "data/processed/features"
    baselines_path: str = "data/processed/eda_baselines.json"
    drift_ks_alpha: float = 0.01       # KS-test significance level for drift alert
    log_to_mlflow: bool = True
    mlflow_run_name: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FeatureEngineeringConfig":
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        obj = cls()
        if "global_descriptors" in raw:
            obj.global_descriptors = GlobalDescriptorSpec(**raw["global_descriptors"])
        if "local_features" in raw:
            obj.local_features = LocalFeatureSpec(**raw["local_features"])
        for k in ("output_dir", "baselines_path", "drift_ks_alpha",
                  "log_to_mlflow", "mlflow_run_name"):
            if k in raw:
                setattr(obj, k, raw[k])
        return obj


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class FeatureEngineeringResult:
    scene_name: str
    global_desc_stats: dict = field(default_factory=dict)
    local_feat_stats: dict = field(default_factory=dict)
    drift_alerts: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def summary(self) -> str:
        lines = [f"Scene: {self.scene_name}"]
        for k, v in self.global_desc_stats.items():
            lines.append(f"  {k}: {v}")
        for k, v in self.local_feat_stats.items():
            lines.append(f"  {k}: {v}")
        if self.drift_alerts:
            lines.append(f"  DRIFT ALERTS: {self.drift_alerts}")
        lines.append(f"  total elapsed: {self.elapsed_sec:.1f}s")
        return "\n".join(lines)


# ── Main class ─────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Extracts global descriptors and local features for all images in a Scene.
    Results are persisted to disk and logged to MLflow.
    """

    def __init__(
        self,
        conf: FeatureEngineeringConfig,
        device: Optional[torch.device] = None,
    ):
        self.conf = conf
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._baselines: Optional[dict] = self._load_baselines()
        logger.info(
            f"FeatureEngineer | device={self.device} | "
            f"global={self._active_global()} | local={self._active_local()}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        scene: Scene,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> FeatureEngineeringResult:
        """Extract all configured features for every image in the scene."""
        t0 = time.perf_counter()
        result = FeatureEngineeringResult(scene_name=scene.scene)

        out_dir = Path(self.conf.output_dir) / scene.dataset / scene.scene
        out_dir.mkdir(parents=True, exist_ok=True)

        mlflow_ctx = (
            mlflow.start_run(
                run_name=self.conf.mlflow_run_name
                or f"features_{scene.scene}",
                nested=True,
            )
            if self.conf.log_to_mlflow
            else _NullCtx()
        )

        with mlflow_ctx:
            if self.conf.log_to_mlflow:
                mlflow.log_params({
                    "scene": scene.scene,
                    "dataset": scene.dataset,
                    "n_images": len(scene.image_paths),
                    "active_global": str(self._active_global()),
                    "active_local": str(self._active_local()),
                    "device": str(self.device),
                })

            # ── A. Global descriptors ─────────────────────────────────────
            global_stats = self._extract_global_descriptors(
                scene, out_dir, progress_bar
            )
            result.global_desc_stats = global_stats

            # ── B. Local features ─────────────────────────────────────────
            local_stats = self._extract_local_features(
                scene, out_dir, progress_bar
            )
            result.local_feat_stats = local_stats

            # ── C. Drift detection ────────────────────────────────────────
            drift_alerts = self._check_drift(global_stats)
            result.drift_alerts = drift_alerts

            result.elapsed_sec = time.perf_counter() - t0

            if self.conf.log_to_mlflow:
                all_metrics: dict = {}
                all_metrics.update(global_stats)
                all_metrics.update(local_stats)
                all_metrics["total_elapsed_sec"] = round(result.elapsed_sec, 2)
                all_metrics["drift_alert_count"] = len(drift_alerts)
                mlflow.log_metrics(all_metrics)
                if drift_alerts:
                    mlflow.set_tag(
                        "drift_alerts", " | ".join(drift_alerts)
                    )

        logger.info(result.summary())
        return result

    # ── Global descriptor extraction ──────────────────────────────────────────

    def _extract_global_descriptors(
        self,
        scene: Scene,
        out_dir: Path,
        progress_bar: Optional[tqdm.tqdm],
    ) -> dict:
        stats: dict = {}
        cfg = self.conf.global_descriptors

        if cfg.dinov2:
            s = self._extract_one_global(scene, out_dir, "dinov2", progress_bar)
            stats.update(s)

        if cfg.isc:
            s = self._extract_one_global(scene, out_dir, "isc", progress_bar)
            stats.update(s)

        if cfg.mast3r_spoc:
            s = self._extract_one_global(scene, out_dir, "mast3r_spoc", progress_bar)
            stats.update(s)

        if cfg.siglip2:
            s = self._extract_one_global(scene, out_dir, "siglip2", progress_bar)
            stats.update(s)

        return stats

    def _extract_one_global(
        self,
        scene: Scene,
        out_dir: Path,
        model_name: str,
        progress_bar: Optional[tqdm.tqdm],
    ) -> dict:
        """
        Build the appropriate extractor, run inference, save to .npy,
        return a dict of scalar stats for MLflow.
        """
        prefix = f"global_{model_name}"
        out_path = out_dir / f"{prefix}.npy"

        if out_path.exists():
            logger.info(f"[{scene.scene}] {prefix}: cache hit, skipping inference")
            feats = np.load(str(out_path))
            return self._desc_stats(feats, prefix)

        t0 = time.perf_counter()
        try:
            extractor = self._build_global_extractor(model_name)
            dataset = extractor.create_dataset_from_scene(scene)
            dl_params = extractor.get_dataloader_params()
            loader = DataLoader(
                dataset,
                batch_size=self.conf.global_descriptors.batch_size,
                shuffle=False,
                num_workers=0,
                **dl_params,
            )

            all_feats: list[torch.Tensor] = []
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    feat = extractor(batch)
                    all_feats.append(feat.cpu().float())
                    if progress_bar:
                        progress_bar.set_postfix_str(
                            f"{prefix} ({i + 1}/{len(loader)})"
                        )

            feats_t = torch.cat(all_feats, dim=0)  # (N, D)
            feats_np = feats_t.numpy()
            np.save(str(out_path), feats_np)
            logger.info(
                f"[{scene.scene}] {prefix}: shape={feats_np.shape}, "
                f"saved to {out_path}, "
                f"elapsed={time.perf_counter()-t0:.1f}s"
            )
            return self._desc_stats(feats_np, prefix)

        except Exception as exc:
            logger.error(
                f"[{scene.scene}] {prefix} extraction failed: {exc}"
            )
            return {f"{prefix}_failed": 1}

    def _build_global_extractor(self, model_name: str):
        """Instantiate the correct extractor class for the given model name."""
        if model_name == "dinov2":
            from global_descriptors.dinov2 import DINOv2GlobalDescriptorExtractor
            from scripts.data import resolve_model_path
            from models.config import HuggingFaceModelConfig
            conf = HuggingFaceModelConfig(pretrained_model="DINOV2_BASE")
            return DINOv2GlobalDescriptorExtractor(conf=conf, device=self.device)

        elif model_name == "isc":
            from global_descriptors.isc import ISCGlobalDescriptorExtractor
            from models.config import ISCModelConfig
            conf = ISCModelConfig(weight_path="ISC")
            return ISCGlobalDescriptorExtractor(conf=conf, device=self.device)

        elif model_name == "mast3r_spoc":
            from global_descriptors.mast3r_spoc import MASt3RSPoCGlobalDescriptorExtractor
            from models.config import MASt3RRetrievalModelConfig, MASt3RModelConfig
            from models.config import HuggingFaceModelConfig
            mconf = MASt3RModelConfig(weight_path="MAST3R")
            conf = MASt3RRetrievalModelConfig(mast3r=mconf)
            return MASt3RSPoCGlobalDescriptorExtractor(conf=conf, device=self.device)

        elif model_name == "siglip2":
            from global_descriptors.siglip2 import SigLIP2GlobalDescriptorExtractor
            from models.config import HuggingFaceModelConfig
            conf = HuggingFaceModelConfig(pretrained_model="SIGLIP2")
            return SigLIP2GlobalDescriptorExtractor(conf=conf, device=self.device)

        else:
            raise ValueError(f"Unknown global descriptor model: {model_name}")

    # ── Local feature extraction ───────────────────────────────────────────────

    def _extract_local_features(
        self,
        scene: Scene,
        out_dir: Path,
        progress_bar: Optional[tqdm.tqdm],
    ) -> dict:
        """
        Extract keypoints + descriptors for every image in the scene.
        Saves one .npz per image per model: {out_dir}/local/{model}/{img_name}.npz
        Each .npz contains: lafs, keypoints, scores, descriptors
        """
        stats: dict = {}
        cfg = self.conf.local_features

        if cfg.aliked:
            s = self._extract_one_local(
                scene, out_dir, "aliked",
                resize=cfg.aliked_resize,
                max_keypoints=cfg.aliked_max_keypoints,
                progress_bar=progress_bar,
            )
            stats.update(s)

        if cfg.superpoint:
            s = self._extract_one_local(
                scene, out_dir, "superpoint",
                resize=cfg.superpoint_resize,
                max_keypoints=cfg.superpoint_max_keypoints,
                progress_bar=progress_bar,
            )
            stats.update(s)

        return stats

    def _extract_one_local(
        self,
        scene: Scene,
        out_dir: Path,
        model_name: str,
        resize: int,
        max_keypoints: int,
        progress_bar: Optional[tqdm.tqdm],
    ) -> dict:
        prefix = f"local_{model_name}"
        model_dir = out_dir / "local" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        kpt_counts: list[int] = []
        n_failed = 0

        try:
            extractor = self._build_local_extractor(model_name, resize, max_keypoints)
        except Exception as exc:
            logger.error(f"[{scene.scene}] {prefix}: extractor build failed: {exc}")
            return {f"{prefix}_failed": 1}

        for i, path in enumerate(scene.image_paths):
            img_name = Path(path).stem
            out_path = model_dir / f"{img_name}.npz"

            if out_path.exists():
                data = np.load(str(out_path))
                kpt_counts.append(len(data["keypoints"]))
                continue

            try:
                lafs, kpts, scores, descs = extractor(
                    path, image_reader=scene.get_image
                )
                np.savez_compressed(
                    str(out_path),
                    lafs=lafs, keypoints=kpts,
                    scores=scores, descriptors=descs,
                )
                kpt_counts.append(len(kpts))
            except Exception as exc:
                logger.warning(f"[{scene.scene}] {prefix} failed on {img_name}: {exc}")
                n_failed += 1
                kpt_counts.append(0)

            if progress_bar:
                progress_bar.set_postfix_str(
                    f"{prefix} ({i + 1}/{len(scene.image_paths)})"
                )

        elapsed = time.perf_counter() - t0
        kpt_arr = np.array(kpt_counts, dtype=float)
        logger.info(
            f"[{scene.scene}] {prefix}: "
            f"mean_kpts={kpt_arr.mean():.0f}, "
            f"failed={n_failed}, elapsed={elapsed:.1f}s"
        )
        return {
            f"{prefix}_mean_kpts": float(kpt_arr.mean()) if len(kpt_arr) else 0.0,
            f"{prefix}_min_kpts":  float(kpt_arr.min()) if len(kpt_arr) else 0.0,
            f"{prefix}_max_kpts":  float(kpt_arr.max()) if len(kpt_arr) else 0.0,
            f"{prefix}_failed":    n_failed,
            f"{prefix}_elapsed_sec": round(elapsed, 2),
        }

    def _build_local_extractor(
        self, model_name: str, resize: int, max_keypoints: int
    ):
        """
        Returns a callable extractor(path, image_reader) → (lafs, kpts, scores, descs).
        Wraps LocalFeatureExtractor with the correct handler config.
        """
        from extractor import LocalFeatureExtractor
        from features.factory import create_local_feature_handler
        from features.config import LocalFeatureConfig
        from preprocesses.config import ResizeConfig

        resize_conf = ResizeConfig(
            func="kornia", long_edge_length=resize
        )

        if model_name == "aliked":
            from models.config import ALIKEDConfig
            handler_conf = LocalFeatureConfig(
                name="aliked",
                aliked=ALIKEDConfig(
                    weight_path="ALIKED_LIGHTGLUE_N16",
                    max_keypoints=max_keypoints,
                ),
                resize=resize_conf,
            )
        elif model_name == "superpoint":
            from models.config import SuperPointConfig
            handler_conf = LocalFeatureConfig(
                name="superpoint",
                superpoint=SuperPointConfig(
                    weight_path="MAGICLEAP_SUPERPOINT",
                    max_keypoints=max_keypoints,
                    nms_radius=4,
                    keypoint_threshold=0.0005,
                ),
                resize=resize_conf,
            )
        else:
            raise ValueError(f"Unknown local feature model: {model_name}")

        handler = create_local_feature_handler(handler_conf, device=self.device)
        return LocalFeatureExtractor(conf=handler_conf, handler=handler)

    # ── Drift detection ────────────────────────────────────────────────────────

    def _check_drift(self, global_stats: dict) -> list[str]:
        """
        Compare extracted descriptor norms against EDA baselines.
        Raises a string alert for each detected drift.
        Stored as MLflow tags so Grafana can pick them up.
        """
        if self._baselines is None:
            return []

        alerts: list[str] = []
        baseline_desc = self._baselines.get("descriptor", {})

        if not baseline_desc:
            return []

        baseline_mean = baseline_desc.get("norm_mean")
        baseline_std  = baseline_desc.get("norm_std")
        baseline_p10  = baseline_desc.get("norm_p10")
        baseline_p90  = baseline_desc.get("norm_p90")

        for key in ("global_dinov2_norm_mean", "global_isc_norm_mean"):
            live_val = global_stats.get(key)
            if live_val is None or baseline_mean is None:
                continue

            # Simple z-score check: alert if > 3 sigma from baseline mean
            if baseline_std and baseline_std > 0:
                z = abs(live_val - baseline_mean) / baseline_std
                if z > 3.0:
                    msg = (
                        f"DRIFT [{key}]: live={live_val:.4f}, "
                        f"baseline_mean={baseline_mean:.4f}, z={z:.2f}"
                    )
                    alerts.append(msg)
                    logger.warning(msg)

        return alerts

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _desc_stats(feats: np.ndarray, prefix: str) -> dict:
        """Compute scalar stats from a (N, D) descriptor matrix."""
        norms = np.linalg.norm(feats, axis=1)
        return {
            f"{prefix}_n_images":    int(feats.shape[0]),
            f"{prefix}_dim":         int(feats.shape[1]),
            f"{prefix}_norm_mean":   float(norms.mean()),
            f"{prefix}_norm_std":    float(norms.std()),
            f"{prefix}_norm_p10":    float(np.percentile(norms, 10)),
            f"{prefix}_norm_p90":    float(np.percentile(norms, 90)),
        }

    def _active_global(self) -> list[str]:
        cfg = self.conf.global_descriptors
        return [m for m, on in [
            ("dinov2", cfg.dinov2), ("isc", cfg.isc),
            ("mast3r_spoc", cfg.mast3r_spoc), ("siglip2", cfg.siglip2),
        ] if on]

    def _active_local(self) -> list[str]:
        cfg = self.conf.local_features
        return [m for m, on in [
            ("aliked", cfg.aliked), ("superpoint", cfg.superpoint),
        ] if on]

    def _load_baselines(self) -> Optional[dict]:
        p = Path(self.conf.baselines_path)
        if p.exists():
            with open(p) as f:
                return json.load(f)
        logger.warning(
            f"EDA baselines not found at {p}. "
            "Run the EDA notebook first to generate drift baselines."
        )
        return None


# ── Null context manager ───────────────────────────────────────────────────────

class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *_): pass
