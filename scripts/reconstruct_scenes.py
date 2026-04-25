"""
scripts/reconstruct_scenes.py
────────────────────────────────────────────────────────────────────────────
DVC Stage: run_pipeline

Main reconstruction entry point. A single invocation of IMC2025Pipeline.run().

Loads real models (MASt3R, ALIKED, DINOv2, etc.) once, then runs
COLMAP-based SfM for all scenes in the train datasets.

Reads:
  - conf/mast3r.yaml                   (pipeline config)
  - data/prepared/prepared_input.csv    (submission-format DataFrame)
  - data/processed/images/              (preprocessed image tree)

Writes:
  - data/reconstruction/eval_prediction.csv       (IMC2025 poses)
  - data/reconstruction/sparse_reconstruction.ply  (best COLMAP model)
  - data/reconstruction/reconstruction_metrics.json (DVC metric file)

MLflow:
  - Child run "run_pipeline" under parent DVC run (via MLFLOW_PARENT_RUN_ID)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.config import PipelineConfig, SubmissionConfig
from scripts.data import DEFAULT_DATASET_DIR, IMC2025TrainData, setup_data_schema
from scripts.distributed import DistConfig
from scripts.pipeline import create_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real IMC2025Pipeline reconstruction for DVC stage"
    )
    parser.add_argument(
        "--config",
        default="conf/mast3r.yaml",
        help="Path to pipeline YAML config",
    )
    parser.add_argument(
        "--prepared-input",
        default="data/prepared/prepared_input.csv",
        help="Path to prepared submission CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="data/reconstruction",
        help="Directory for reconstruction outputs",
    )
    parser.add_argument(
        "--data-root-dir",
        default=None,
        help="Root directory for images (default: data/processed/images if exists, else data/)",
    )
    parser.add_argument(
        "--experiment-name",
        default="scene_reconstruction_dvc",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def flatten_dict(d: dict, prefix: str = "", sep: str = ".") -> dict:
    """Flatten nested dict to dotted-key dict for MLflow param logging."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=key, sep=sep))
        else:
            out[key] = str(v)
    return out


def compute_registration_rate(df: pd.DataFrame) -> tuple[float, int]:
    """Compute registration rate and count of registered images."""
    def is_valid(r_str: str) -> bool:
        try:
            vals = [float(x) for x in str(r_str).split(";")]
            return len(vals) == 9 and not any(v != v for v in vals)
        except Exception:
            return False

    registered = df["rotation_matrix"].apply(is_valid).sum()
    rate = float(registered / max(len(df), 1))
    return rate, int(registered)


def _write_ply_from_poses(ply_path: Path, submission_df: pd.DataFrame) -> None:
    """
    Fallback: generate a PLY file from the translation vectors in the
    submission DataFrame. Each registered image becomes a 3D point.
    """
    points = []
    for _, row in submission_df.iterrows():
        try:
            t_vals = [float(x) for x in str(row["translation_vector"]).split(";")]
            if len(t_vals) == 3 and not any(v != v for v in t_vals):
                points.append(t_vals)
        except Exception:
            continue

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment Generated from {len(points)} camera positions\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i, (x, y, z) in enumerate(points):
            r = (i * 29) % 255
            g = (i * 47) % 255
            b = (i * 61) % 255
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    log.info("Wrote fallback PLY with %d points to %s", len(points), ply_path)


def _try_export_colmap_ply(output_dir: Path) -> Optional[Path]:
    """
    Search for COLMAP reconstruction directories (containing points3D.bin)
    and export the best one as a PLY file.
    Uses pycolmap.Reconstruction().export_PLY().
    """
    try:
        import pycolmap
    except ImportError:
        log.warning("pycolmap not available — cannot export COLMAP PLY")
        return None

    max_pts = -1
    best_rec_dir = None

    # Search in the scene workspace for any COLMAP outputs
    # When SCENE_SPACE_DIR_PERSISTENT=yes, reconstruction dirs persist
    search_dirs = [
        Path("extra/tmp"),  # DEFAULT_TMP_DIR
        output_dir,
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pts_file in search_dir.rglob("points3D.bin"):
            try:
                rec = pycolmap.Reconstruction(str(pts_file.parent))
                n_pts = len(rec.points3D)
                if n_pts > max_pts:
                    max_pts = n_pts
                    best_rec_dir = pts_file.parent
            except Exception:
                pass

    if best_rec_dir is None:
        log.info("No COLMAP reconstruction directories found for PLY export")
        return None

    ply_path = output_dir / "sparse_reconstruction.ply"
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    pycolmap.Reconstruction(str(best_rec_dir)).export_PLY(str(ply_path))
    log.info(
        "Exported COLMAP PLY from %s (%d 3D points) → %s",
        best_rec_dir, max_pts, ply_path,
    )
    return ply_path


def _cleanup_gpu() -> None:
    """Free CUDA cache after reconstruction."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    prepared_input_path = Path(args.prepared_input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not prepared_input_path.exists():
        raise FileNotFoundError(f"Prepared input not found: {prepared_input_path}")

    # ── Determine data root ──────────────────────────────────────────────
    # Preprocessed images live under data/processed/images/train/{dataset}/{image}
    # IMC2025TrainData.build_image_relative_path returns train/{dataset}/{image}
    # So data_root_dir should be data/processed/images
    processed_images_dir = Path("data/processed/images")
    if args.data_root_dir:
        data_root_dir = Path(args.data_root_dir)
    elif processed_images_dir.exists():
        data_root_dir = processed_images_dir
        log.info("Using preprocessed images from %s", data_root_dir)
    else:
        data_root_dir = DEFAULT_DATASET_DIR
        log.info("Falling back to raw data dir %s", data_root_dir)

    # ── Load config ──────────────────────────────────────────────────────
    pipeline_conf = PipelineConfig.load_config(config_path)
    with open(config_path) as f:
        raw_yaml = yaml.safe_load(f)

    log.info("Pipeline type: %s", pipeline_conf.type)

    # ── Set up environment for persistent scene workspace ────────────────
    # This allows us to access COLMAP reconstruction dirs for PLY export
    os.environ["SCENE_SPACE_DIR_PERSISTENT"] = "yes"

    # ── Build DataSchema ─────────────────────────────────────────────────
    submission_conf = SubmissionConfig(
        pipeline=pipeline_conf,
        target_data_type="imc2025train",
    )
    data_schema = setup_data_schema(submission_conf, data_root_dir=data_root_dir)

    # ── Initialize pipeline (loads all models to GPU once) ───────────────
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info("Initializing pipeline on device: %s", device)

    t_init_start = time.perf_counter()
    pipeline = create_pipeline(
        conf=pipeline_conf,
        dist_conf=DistConfig.single(),
        device=device,
    )
    t_init = time.perf_counter() - t_init_start
    log.info("Pipeline initialized in %.1fs", t_init)

    # ── Run pipeline ─────────────────────────────────────────────────────
    log.info("Starting reconstruction for %d images...", len(data_schema.df))

    t_run_start = time.perf_counter()
    submission_df = pipeline.run(
        df=data_schema.df,
        data_schema=data_schema,
        save_snapshot=True,
    )
    t_run = time.perf_counter() - t_run_start
    log.info("Pipeline completed in %.1fs", t_run)

    # ── Compute metrics ──────────────────────────────────────────────────
    reg_rate, n_registered = compute_registration_rate(submission_df)
    total_time = t_init + t_run
    n_scenes = submission_df.groupby(["dataset", "scene"]).ngroups if "scene" in submission_df.columns else 0
    n_datasets = submission_df["dataset"].nunique() if "dataset" in submission_df.columns else 0

    log.info(
        "Registration: %d/%d (%.1f%%)  Scenes: %d  Time: %.1fs",
        n_registered, len(submission_df), 100 * reg_rate, n_scenes, total_time,
    )

    # ── Save eval_prediction.csv ─────────────────────────────────────────
    eval_csv_path = output_dir / "eval_prediction.csv"
    submission_df.to_csv(eval_csv_path, index=False)
    log.info("Saved eval_prediction.csv → %s", eval_csv_path)

    # ── Export PLY ────────────────────────────────────────────────────────
    ply_path = _try_export_colmap_ply(output_dir)
    if ply_path is None:
        # Fallback: generate PLY from camera positions
        ply_path = output_dir / "sparse_reconstruction.ply"
        _write_ply_from_poses(ply_path, submission_df)

    # ── Save reconstruction metrics JSON (DVC metrics file) ──────────────
    metrics = {
        "registration_rate": round(reg_rate, 4),
        "n_registered": n_registered,
        "total_images": len(submission_df),
        "n_scenes": n_scenes,
        "n_datasets": n_datasets,
        "init_time_seconds": round(t_init, 2),
        "run_time_seconds": round(t_run, 2),
        "total_time_seconds": round(total_time, 2),
        "device": str(device),
    }

    metrics_path = output_dir / "reconstruction_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved reconstruction_metrics.json → %s", metrics_path)

    # ── MLflow child run ─────────────────────────────────────────────────
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")
    run_tags = {"mlflow.parentRunId": parent_run_id} if parent_run_id else None

    # Log flattened config as params
    flat_params = flatten_dict(raw_yaml)
    flat_params = {k: v[:500] for k, v in flat_params.items()}

    with mlflow.start_run(run_name="run_pipeline", nested=True, tags=run_tags):
        # Params
        mlflow.log_params(flat_params)
        mlflow.log_param("device", str(device))
        mlflow.log_param("config_file", config_path.name)
        mlflow.log_param("data_root_dir", str(data_root_dir))
        mlflow.log_param("pipeline_type", pipeline_conf.type)

        if torch.cuda.is_available():
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
            props = torch.cuda.get_device_properties(0)
            mlflow.log_param("gpu_vram_gb", f"{props.total_memory / 1024**3:.1f}")

        # Metrics
        mlflow.log_metric("registration_rate", round(reg_rate, 4))
        mlflow.log_metric("n_registered", n_registered)
        mlflow.log_metric("total_images", len(submission_df))
        mlflow.log_metric("n_scenes", n_scenes)
        mlflow.log_metric("n_datasets", n_datasets)
        mlflow.log_metric("init_time_seconds", round(t_init, 2))
        mlflow.log_metric("run_time_seconds", round(t_run, 2))
        mlflow.log_metric("total_time_seconds", round(total_time, 2))

        # Artifacts
        mlflow.log_artifact(str(eval_csv_path), artifact_path="reconstruction")
        if ply_path and ply_path.exists():
            mlflow.log_artifact(str(ply_path), artifact_path="reconstruction")
        mlflow.log_artifact(str(config_path), artifact_path="config")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

    log.info("MLflow child run 'run_pipeline' logged successfully")

    # ── Cleanup ──────────────────────────────────────────────────────────
    _cleanup_gpu()


if __name__ == "__main__":
    main()
