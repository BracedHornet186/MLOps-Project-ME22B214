"""
scripts/train_experiment.py
────────────────────────────────────────────────────────────────────────────
Stage 3 — Model Development & Experiment Tracking

Wraps evaluate_imc2025.py with full MLflow instrumentation:
  - Logs all pipeline config params (flattened from YAML)
  - Logs per-dataset and overall mAA metrics
  - Logs latency, registration rate, inlier ratio
  - Saves config YAML + submission CSV as artifacts
  - Registers the best config in the MLflow Model Registry
  - Compares against previous runs and prints a leaderboard

Usage:
    python3 scripts/train_experiment.py \
        --config conf/pipeline/imc2025/mast3r_rtx3060.yaml \
        --datasets ETs stairs \
        --experiment-name scene_reconstruction \
        --run-name mast3r_rtx3060_v1

    # Run multiple configs and compare:
    python3 scripts/train_experiment.py \
        --config conf/pipeline/imc2025/mast3r_rtx3060.yaml \
        --compare-with conf/pipeline/imc2025/mast3r_rtx3060_v2.yaml \
        --datasets ETs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
import torch
import yaml

# ── Project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.config import PipelineConfig, SubmissionConfig
from scripts.data import DEFAULT_DATASET_DIR, IMC2025TrainData, setup_data_schema
import utils.imc25.metric as imc25_metric

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_git_commit() -> str:
    """Return current short git commit hash, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


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


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_gpu_info() -> dict:
    """Return basic GPU info for tagging the MLflow run."""
    info = {"gpu_available": str(torch.cuda.is_available())}
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_vram_gb"] = f"{props.total_memory / 1024**3:.1f}"
    return info


def registration_rate(submission_df: pd.DataFrame) -> float:
    """Fraction of images with non-NaN rotation matrix."""
    def is_valid(r_str: str) -> bool:
        try:
            vals = [float(x) for x in str(r_str).split(";")]
            return len(vals) == 9 and not any(
                v != v for v in vals  # NaN check
            )
        except Exception:
            return False

    valid = submission_df["rotation_matrix"].apply(is_valid).sum()
    return float(valid / max(len(submission_df), 1))


def run_pipeline_and_collect_metrics(
    conf: SubmissionConfig,
    datasets: Optional[list[str]],
    submission_csv: Path,
) -> tuple[float, dict, float]:
    """
    Run the full pipeline on the given datasets.
    Returns (overall_mAA, per_dataset_scores, wall_clock_seconds).
    """
    from scripts.kernel import run_and_save_submission
    from scripts.distributed import DistConfig

    if datasets:
        conf.datasets_to_use = datasets

    t_start = time.perf_counter()
    run_and_save_submission(
        conf,
        env_name="local",
        data_root_dir=DEFAULT_DATASET_DIR,
        dist_conf=DistConfig.single(),
    )
    elapsed = time.perf_counter() - t_start

    assert submission_csv.exists(), "Pipeline finished but submission.csv not found"

    # ── Score against ground truth ─────────────────────────────────────────
    user_df = pd.read_csv(submission_csv)
    gt_df = IMC2025TrainData.create(DEFAULT_DATASET_DIR).df

    final_score, dataset_scores = imc25_metric.score(
        gt_csv=DEFAULT_DATASET_DIR / "train_labels.csv",
        user_csv=submission_csv,
        thresholds_csv=DEFAULT_DATASET_DIR / "train_thresholds.csv",
        mask_csv=None,
        inl_cf=0,
        strict_cf=-1,
        verbose=True,
    )

    per_dataset = {ds: float(sc) for ds, sc in dataset_scores.items()}
    return float(final_score), per_dataset, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    config_path: str,
    datasets: Optional[list[str]],
    experiment_name: str,
    run_name: str,
    mlflow_uri: str,
    register_if_best: bool = True,
) -> str:
    """
    Execute one pipeline config, log everything to MLflow, return the run_id.
    """
    config_path = Path(config_path)
    submission_csv = ROOT / "submission.csv"

    # ── MLflow setup ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    pipeline_conf = PipelineConfig.load_config(config_path)
    submission_conf = SubmissionConfig(
        pipeline=pipeline_conf,
        target_data_type="imc2025train",
    )

    raw_yaml = load_yaml(config_path)
    flat_params = flatten_dict(raw_yaml)
    # MLflow param values must be strings ≤ 500 chars
    flat_params = {k: v[:500] for k, v in flat_params.items()}

    git_commit = get_git_commit()
    gpu_info = compute_gpu_info()

    log.info("Starting MLflow run: %s / %s", experiment_name, run_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # ── Tags ──────────────────────────────────────────────────────────
        mlflow.set_tags({
            "git_commit": git_commit,
            "config_file": config_path.name,
            "pipeline_type": raw_yaml.get("type", "unknown"),
            **gpu_info,
        })

        # ── Parameters ────────────────────────────────────────────────────
        mlflow.log_params(flat_params)
        if datasets:
            mlflow.log_param("datasets_evaluated", ",".join(datasets))

        # ── Run pipeline ──────────────────────────────────────────────────
        log.info("Running inference pipeline...")
        try:
            overall_maa, per_dataset, elapsed = run_pipeline_and_collect_metrics(
                submission_conf, datasets, submission_csv
            )
            pipeline_success = True
        except Exception as e:
            log.error("Pipeline failed: %s", e)
            mlflow.set_tag("pipeline_status", "FAILED")
            mlflow.log_param("failure_reason", str(e)[:500])
            raise

        # ── Core metrics ──────────────────────────────────────────────────
        mlflow.log_metric("mAA_overall", overall_maa)
        for ds, score in per_dataset.items():
            mlflow.log_metric(f"mAA_{ds}", score)

        # ── Operational metrics ───────────────────────────────────────────
        submission_df = pd.read_csv(submission_csv)
        reg_rate = registration_rate(submission_df)
        n_scenes = submission_df.groupby(["dataset", "scene"]).ngroups if "scene" in submission_df.columns else 0

        mlflow.log_metrics({
            "inference_latency_seconds": round(elapsed, 2),
            "registration_rate": round(reg_rate, 4),
            "total_images": len(submission_df),
            "n_scenes": n_scenes,
            "latency_per_scene_seconds": round(elapsed / max(n_scenes, 1), 2),
        })
        mlflow.set_tag("pipeline_status", "SUCCESS")

        # ── Artifacts ─────────────────────────────────────────────────────
        mlflow.log_artifact(str(config_path), artifact_path="config")
        mlflow.log_artifact(str(submission_csv), artifact_path="predictions")

        # Write per-dataset JSON for easy downstream comparison
        per_dataset_path = ROOT / "per_dataset_scores.json"
        per_dataset_path.write_text(json.dumps(per_dataset, indent=2))
        mlflow.log_artifact(str(per_dataset_path), artifact_path="metrics")

        # ── MLflow Model Registry ─────────────────────────────────────────
        # We register the config YAML as a "model" artifact so it can be
        # transitioned through Staging → Production in the Registry UI.
        if register_if_best:
            _maybe_register_model(
                run_id=run_id,
                model_name=f"scene_reconstruction_{config_path.stem}",
                overall_maa=overall_maa,
                experiment_name=experiment_name,
                mlflow_uri=mlflow_uri,
            )

        log.info(
            "Run complete — mAA=%.4f  reg_rate=%.2f%%  elapsed=%.1fs  run_id=%s",
            overall_maa, 100 * reg_rate, elapsed, run_id,
        )

    return run_id


def _maybe_register_model(
    run_id: str,
    model_name: str,
    overall_maa: float,
    experiment_name: str,
    mlflow_uri: str,
) -> None:
    """Register run in Model Registry if it beats the current Production model."""
    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)

    # Check if a Production version already exists
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            prod_run_id = prod_versions[0].run_id
            prod_run = client.get_run(prod_run_id)
            prod_maa = prod_run.data.metrics.get("mAA_overall", 0.0)
            if overall_maa <= prod_maa:
                log.info(
                    "New mAA (%.4f) does not beat Production (%.4f). Registering to Staging only.",
                    overall_maa, prod_maa,
                )
                stage = "Staging"
            else:
                log.info(
                    "New mAA (%.4f) beats Production (%.4f). Promoting to Production.",
                    overall_maa, prod_maa,
                )
                stage = "Production"
        else:
            stage = "Production"  # first run always goes to Production
    except mlflow.exceptions.RestException:
        stage = "Production"

    # Register the run
    model_uri = f"runs:/{run_id}/config"
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)

    client.transition_model_version_stage(
        name=model_name,
        version=registered.version,
        stage=stage,
    )
    log.info("Registered model '%s' v%s → %s", model_name, registered.version, stage)


# ─────────────────────────────────────────────────────────────────────────────
# Leaderboard across all runs of an experiment
# ─────────────────────────────────────────────────────────────────────────────

def print_leaderboard(experiment_name: str, mlflow_uri: str, top_n: int = 10) -> None:
    """Print a ranked table of all runs in the experiment."""
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        log.warning("Experiment '%s' not found.", experiment_name)
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.mAA_overall DESC"],
        max_results=top_n,
    )

    if not runs:
        log.info("No finished runs found in experiment '%s'.", experiment_name)
        return

    rows = []
    for r in runs:
        rows.append({
            "run_name":  r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
            "mAA":       f"{r.data.metrics.get('mAA_overall', 0):.4f}",
            "reg_rate":  f"{100*r.data.metrics.get('registration_rate', 0):.1f}%",
            "latency_s": f"{r.data.metrics.get('inference_latency_seconds', 0):.0f}s",
            "config":    r.data.tags.get("config_file", "—"),
            "git":       r.data.tags.get("git_commit", "—"),
            "run_id":    r.info.run_id[:8],
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print(f"  Leaderboard — {experiment_name}  (top {len(runs)})")
    print("=" * 70)
    print(df.to_string(index=True))
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run scene reconstruction pipeline with MLflow experiment tracking"
    )
    p.add_argument("-c", "--config", required=True,
                   help="Path to pipeline YAML config")
    p.add_argument("--compare-with", nargs="*", default=[],
                   help="Additional configs to run and compare against")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Dataset names to evaluate on (default: all train datasets)")
    p.add_argument("--experiment-name", default="scene_reconstruction",
                   help="MLflow experiment name")
    p.add_argument("--run-name", default=None,
                   help="MLflow run name (default: config filename stem)")
    p.add_argument("--mlflow-uri",
                   default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
                   help="MLflow tracking server URI")
    p.add_argument("--leaderboard-only", action="store_true",
                   help="Only print the leaderboard, skip running inference")
    p.add_argument("--no-register", action="store_false", dest="register",
                   help="Skip MLflow Model Registry registration")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.leaderboard_only:
        print_leaderboard(args.experiment_name, args.mlflow_uri)
        return

    all_configs = [args.config] + list(args.compare_with)
    run_ids = []

    for cfg_path in all_configs:
        run_name = args.run_name or Path(cfg_path).stem
        if len(all_configs) > 1:
            run_name = Path(cfg_path).stem  # unique name per config when comparing

        run_id = run_experiment(
            config_path=cfg_path,
            datasets=args.datasets,
            experiment_name=args.experiment_name,
            run_name=run_name,
            mlflow_uri=args.mlflow_uri,
            register_if_best=args.register,
        )
        run_ids.append((cfg_path, run_id))

    if len(run_ids) > 1:
        log.info("Comparison run_ids:")
        for cfg, rid in run_ids:
            log.info("  %-50s → %s", cfg, rid)

    print_leaderboard(args.experiment_name, args.mlflow_uri)


if __name__ == "__main__":
    main()
