from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config import load_pipeline_config
from data import DEFAULT_DATASET_DIR
import utils.imc25.metric as imc25_metric

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reconstruction outputs and log DVC metrics")
    parser.add_argument("--reconstruction-dir", default="data/reconstruction")
    parser.add_argument("--config", default="conf/mast3r.yaml")
    parser.add_argument("--metrics-dir", default="data/metrics")
    parser.add_argument("--experiment-name", default="scene_reconstruction_dvc")
    parser.add_argument("--run-name", default="evaluate")
    parser.add_argument(
        "--eval-prediction-csv",
        default=None,
        help="Path to eval_prediction.csv from run_pipeline stage (skips legacy build)",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    return parser.parse_args()


def _flatten_dict(d: dict, prefix: str = "", sep: str = ".") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=key, sep=sep))
        else:
            out[key] = str(v)
    return out


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _discover_scene_summaries(reconstruction_dir: Path) -> list[tuple[str, str]]:
    """Find all (dataset, scene) pairs that have a reconstruction_summary.json."""
    scenes: list[tuple[str, str]] = []
    if not reconstruction_dir.exists():
        return scenes
    for ds_dir in sorted(reconstruction_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        for scene_dir in sorted(ds_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            if (scene_dir / "reconstruction_summary.json").exists():
                scenes.append((ds_dir.name, scene_dir.name))
    return scenes


def _collect_reconstruction_stats(reconstruction_dir: Path) -> tuple[int, int]:
    """Sum reconstruction_points and registered_images across all scenes."""
    total_points = 0
    total_registered = 0

    for ds_dir in sorted(reconstruction_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        for scene_dir in sorted(ds_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            summary = _safe_read_json(scene_dir / "reconstruction_summary.json")
            total_points += int(summary.get("reconstruction_points", 0))

            registered = set()
            for cluster in summary.get("clusters", []):
                if isinstance(cluster, list):
                    registered.update(cluster)
            for row in summary.get("cluster_labels", []):
                registered.update(str(x) for x in row.get("images", []))
            total_registered += len(registered)

    return total_points, total_registered


def _collect_all_poses(reconstruction_dir: Path) -> dict[str, dict[str, str]]:
    """Collect image_id -> pose from all per-scene reconstruction summaries."""
    all_poses: dict[str, dict[str, str]] = {}
    for ds_dir in sorted(reconstruction_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        for scene_dir in sorted(ds_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            summary = _safe_read_json(scene_dir / "reconstruction_summary.json")
            all_poses.update(summary.get("poses", {}))
    return all_poses


def _collect_reconstructed_ids(reconstruction_dir: Path) -> set[str]:
    """Collect all image_ids that appear in any cluster across scenes."""
    ids: set[str] = set()
    for ds_dir in sorted(reconstruction_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        for scene_dir in sorted(ds_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            summary = _safe_read_json(scene_dir / "reconstruction_summary.json")
            for cluster in summary.get("clusters", []):
                if isinstance(cluster, list):
                    ids.update(str(x) for x in cluster)
            for row in summary.get("cluster_labels", []):
                ids.update(str(x) for x in row.get("images", []))
    return ids


def _read_id_to_raw_path(extracted_dir: Path) -> dict[str, Path]:
    """Build image_id -> raw_path from per-scene extracted indexes."""
    id_to_path: dict[str, Path] = {}
    if not extracted_dir.exists():
        return id_to_path
    for ds_dir in sorted(extracted_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        for scene_dir in sorted(ds_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            payload = _safe_read_json(scene_dir / "extracted_index.json")
            for rec in payload.get("records", []):
                image_id = rec.get("image_id")
                raw_path = rec.get("raw_path") or rec.get("source_path")
                if image_id and raw_path:
                    id_to_path[str(image_id)] = Path(str(raw_path))
    return id_to_path


def _build_eval_prediction_csv(
    reconstruction_dir: Path,
    extracted_dir: Path,
    out_csv: Path,
) -> tuple[Path, float]:
    import hashlib

    gt_csv = DEFAULT_DATASET_DIR / "train_labels.csv"
    gt_df = pd.read_csv(gt_csv)

    reconstructed_ids = _collect_reconstructed_ids(reconstruction_dir)
    id_to_raw_path = _read_id_to_raw_path(extracted_dir)
    stored_poses = _collect_all_poses(reconstruction_dir)

    inv_map: dict[tuple, str] = {}
    for image_id, raw_path in id_to_raw_path.items():
        parts = raw_path.parts
        if len(parts) >= 2:
            inv_map[(parts[-2], parts[-1])] = image_id
        if len(parts) >= 3:
            inv_map[(parts[-3], parts[-2], parts[-1])] = image_id

    nan_r = ";".join(["nan"] * 9)
    nan_t = ";".join(["nan"] * 3)

    preds = []
    registered = 0
    for row in gt_df.itertuples(index=False):
        dataset = str(row.dataset)
        scene = str(row.scene)
        image = str(row.image)

        image_id = inv_map.get((dataset, scene, image))
        if image_id is None:
            image_id = inv_map.get((dataset, image))

        if image_id is None:
            rel = Path("data") / "train" / dataset / scene / image
            image_id = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()

        if image_id in reconstructed_ids and image_id in stored_poses:
            pose = stored_poses[image_id]
            rotation_matrix = pose.get("rotation_matrix", nan_r)
            translation_vector = pose.get("translation_vector", nan_t)
            registered += 1
        else:
            rotation_matrix = nan_r
            translation_vector = nan_t

        preds.append({
            "dataset": dataset,
            "scene": scene,
            "image": image,
            "rotation_matrix": rotation_matrix,
            "translation_vector": translation_vector,
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_csv(out_csv, index=False)
    registration_rate = float(registered / max(len(preds), 1))
    return out_csv, registration_rate


def _compute_maa(user_csv: Path) -> tuple[float, dict[str, float]]:
    final_scores_tuple, dataset_scores_tuple = imc25_metric.score(
        gt_csv=DEFAULT_DATASET_DIR / "train_labels.csv",
        user_csv=user_csv,
        thresholds_csv=DEFAULT_DATASET_DIR / "train_thresholds.csv",
        mask_csv=None,
        inl_cf=0,
        strict_cf=-1,
        verbose=True,
    )
    final_score = float(final_scores_tuple[0])
    dataset_scores = dataset_scores_tuple[0]
    per_dataset = {str(ds): float(sc) for ds, sc in dataset_scores.items()}
    return final_score, per_dataset


def _find_any_ply(reconstruction_dir: Path, metrics_dir: Path, num_points: int) -> Path:
    """Find any PLY file across scene dirs, or generate a fallback."""
    ply_candidates = sorted(reconstruction_dir.rglob("*.ply"))
    if ply_candidates:
        return ply_candidates[0]

    metrics_dir.mkdir(parents=True, exist_ok=True)
    fallback = metrics_dir / "final.ply"
    fallback.write_text(
        "\n".join([
            "ply",
            "format ascii 1.0",
            f"comment generated_by=evaluate estimated_points={num_points}",
            "element vertex 0",
            "property float x",
            "property float y",
            "property float z",
            "end_header",
            "",
        ])
    )
    return fallback


def _get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _get_git_status() -> str:
    try:
        return (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()

    conf = load_pipeline_config(args.config)
    if conf.pipeline.type != "imc2025" or conf.pipeline.imc2025_pipeline is None:
        raise ValueError(f"Expected an imc2025 pipeline config, got type={conf.pipeline.type}")

    reconstruction_dir = Path(args.reconstruction_dir)
    extracted_dir = ROOT / "data" / "extracted"
    config_path = Path(args.config)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    raw_conf = {
        "type": conf.pipeline.type,
        "imc2025_pipeline": conf.pipeline.imc2025_pipeline.model_dump(exclude_none=True),
    }
    flat_params = {k: v[:500] for k, v in _flatten_dict(raw_conf).items()}

    # ── Check for eval_prediction.csv from run_pipeline stage ────────────
    # The new run_pipeline stage produces eval_prediction.csv directly.
    # If it exists, use it directly instead of building from legacy
    # reconstruction_summary.json + extracted_index.json files.
    pipeline_eval_csv = reconstruction_dir / "eval_prediction.csv"
    if args.eval_prediction_csv:
        pipeline_eval_csv = Path(args.eval_prediction_csv)

    if pipeline_eval_csv.exists():
        log.info("Using eval_prediction.csv from run_pipeline: %s", pipeline_eval_csv)
        eval_prediction_csv = metrics_dir / "eval_predictions.csv"
        eval_prediction_csv.parent.mkdir(parents=True, exist_ok=True)

        # Copy to metrics dir for consistency
        import shutil
        shutil.copy(pipeline_eval_csv, eval_prediction_csv)

        # Compute registration rate from the CSV
        pred_df = pd.read_csv(eval_prediction_csv)
        registered = 0
        for _, row in pred_df.iterrows():
            try:
                vals = [float(x) for x in str(row["rotation_matrix"]).split(";")]
                if len(vals) == 9 and not any(v != v for v in vals):
                    registered += 1
            except Exception:
                pass
        registration_rate = float(registered / max(len(pred_df), 1))
        num_images = registered
        num_points = 0  # Not available from CSV-only path
        scenes = list(pred_df.groupby(["dataset", "scene"]).groups.keys()) if "scene" in pred_df.columns else []
        log.info("Registration rate: %.4f (%d/%d)", registration_rate, registered, len(pred_df))
    else:
        # Legacy path: build from reconstruction_summary.json + extracted_index.json
        log.info("No eval_prediction.csv found, building from reconstruction summaries")
        if not reconstruction_dir.exists():
            raise FileNotFoundError(f"Reconstruction directory not found: {reconstruction_dir}")

        scenes = _discover_scene_summaries(reconstruction_dir)
        if not scenes:
            raise FileNotFoundError(f"No per-scene reconstruction summaries found under {reconstruction_dir}")
        log.info("Found %d scenes for evaluation", len(scenes))

        num_points, num_images = _collect_reconstruction_stats(reconstruction_dir)
        eval_prediction_csv = metrics_dir / "eval_predictions.csv"
        eval_prediction_csv, registration_rate = _build_eval_prediction_csv(
            reconstruction_dir=reconstruction_dir,
            extracted_dir=extracted_dir,
            out_csv=eval_prediction_csv,
        )

    maa, per_dataset = _compute_maa(eval_prediction_csv)

    metrics = {
        "mAA_overall": float(maa),
        "num_points": int(num_points),
        "num_images": int(num_images),
        "registration_rate": float(registration_rate),
        "num_scenes": len(scenes),
    }

    metrics_path = metrics_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    final_ply_path = _find_any_ply(reconstruction_dir, metrics_dir, num_points)

    git_commit = _get_git_commit()
    git_status = _get_git_status()
    git_status_path = metrics_dir / "git_status.txt"
    git_status_path.write_text(git_status + "\n")

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")
    run_tags = {"mlflow.parentRunId": parent_run_id} if parent_run_id else None

    with mlflow.start_run(run_name=args.run_name, nested=True, tags=run_tags):
        if flat_params:
            mlflow.log_params(flat_params)

        mlflow.log_metric("mAA_overall", float(maa))
        mlflow.log_metric("num_points", int(num_points))
        mlflow.log_metric("num_images", int(num_images))
        mlflow.log_metric("num_scenes", len(scenes))
        mlflow.log_metric("registration_rate", float(registration_rate))
        for ds, score in per_dataset.items():
            mlflow.log_metric(f"mAA_{ds}", score)

        mlflow.set_tag("git_commit", git_commit)
        mlflow.log_artifact(str(git_status_path), artifact_path="git")

        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(eval_prediction_csv), artifact_path="metrics")
        mlflow.log_artifacts(str(reconstruction_dir), artifact_path="reconstruction")

        if final_ply_path.exists():
            mlflow.log_artifact(str(final_ply_path), artifact_path="reconstruction")

        if config_path.exists():
            mlflow.log_artifact(str(config_path), artifact_path="config")

    if parent_run_id:
        client = mlflow.tracking.MlflowClient()
        client.log_metric(parent_run_id, "mAA_overall", float(maa))
        client.log_metric(parent_run_id, "num_points", int(num_points))
        client.log_metric(parent_run_id, "num_images", int(num_images))
        client.log_metric(parent_run_id, "registration_rate", float(registration_rate))
        for ds, score in per_dataset.items():
            client.log_metric(parent_run_id, f"mAA_{ds}", score)
        if config_path.exists():
            client.log_artifact(parent_run_id, str(config_path), artifact_path="config")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
