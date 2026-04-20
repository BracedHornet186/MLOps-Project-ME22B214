from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import hashlib
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
    parser.add_argument("--reconstruction-dir", default="data/reconstruction", help="Input reconstruction directory")
    parser.add_argument("--config", default="conf/mast3r.yaml", help="Pipeline config YAML used for the run")
    parser.add_argument("--metrics-dir", default="data/metrics", help="Output directory for evaluation metrics")
    parser.add_argument("--experiment-name", default="scene_reconstruction_dvc", help="MLflow experiment name")
    parser.add_argument("--run-name", default="evaluate", help="MLflow run name")
    parser.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking server URI",
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


def _extract_reconstruction_stats(reconstruction_dir: Path) -> tuple[int, int]:
    summary = _safe_read_json(reconstruction_dir / "reconstruction_summary.json")

    num_points = int(summary.get("reconstruction_points", 0))

    clusters = summary.get("clusters", [])
    registered_images = len(
        {
            img
            for cluster in clusters
            if isinstance(cluster, list)
            for img in cluster
        }
    )

    if registered_images == 0:
        cluster_labels = summary.get("cluster_labels", [])
        registered_images = len(
            {
                img
                for row in cluster_labels
                for img in row.get("images", [])
            }
        )

    if num_points == 0:
        points_path = reconstruction_dir / "points3d.txt"
        if points_path.exists():
            for line in points_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 2 and parts[0] == "reconstruction_points":
                    try:
                        num_points = int(parts[1])
                    except ValueError:
                        pass

    return num_points, registered_images


def _read_reconstructed_ids(reconstruction_dir: Path) -> set[str]:
    summary = _safe_read_json(reconstruction_dir / "reconstruction_summary.json")
    ids: set[str] = set()

    for cluster in summary.get("clusters", []):
        if isinstance(cluster, list):
            ids.update(str(x) for x in cluster)

    for row in summary.get("cluster_labels", []):
        ids.update(str(x) for x in row.get("images", []))

    return ids


def _read_id_to_raw_path() -> dict[str, Path]:
    extracted_index_path = ROOT / "data" / "extracted" / "extracted_index.json"
    payload = _safe_read_json(extracted_index_path)
    id_to_path: dict[str, Path] = {}

    for rec in payload.get("records", []):
        image_id = rec.get("image_id")
        raw_path = rec.get("raw_path") or rec.get("source_path")
        if image_id and raw_path:
            id_to_path[str(image_id)] = Path(str(raw_path))

    return id_to_path


def _build_eval_prediction_csv(reconstruction_dir: Path, out_csv: Path) -> tuple[Path, float]:
    gt_csv = DEFAULT_DATASET_DIR / "train_labels.csv"
    gt_df = pd.read_csv(gt_csv)

    reconstructed_ids = _read_reconstructed_ids(reconstruction_dir)
    id_to_raw_path = _read_id_to_raw_path()

    # Read stored poses from reconstruction summary
    summary = _safe_read_json(reconstruction_dir / "reconstruction_summary.json")
    stored_poses: dict[str, dict[str, str]] = summary.get("poses", {})

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

        preds.append(
            {
                "dataset": dataset,
                "scene": scene,
                "image": image,
                "rotation_matrix": rotation_matrix,
                "translation_vector": translation_vector,
            }
        )

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


def _ensure_final_ply(reconstruction_dir: Path, metrics_dir: Path, num_points: int) -> Path:
    ply_candidates = sorted(reconstruction_dir.rglob("*.ply"))
    if ply_candidates:
        return ply_candidates[0]

    metrics_dir.mkdir(parents=True, exist_ok=True)
    fallback = metrics_dir / "final.ply"
    fallback.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                f"comment generated_by=evaluate estimated_points={num_points}",
                "element vertex 0",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "",
            ]
        )
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
    config_path = Path(args.config)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not reconstruction_dir.exists():
        raise FileNotFoundError(f"Reconstruction directory not found: {reconstruction_dir}")

    raw_conf = {
        "type": conf.pipeline.type,
        "imc2025_pipeline": conf.pipeline.imc2025_pipeline.model_dump(exclude_none=True),
    }
    flat_params = {k: v[:500] for k, v in _flatten_dict(raw_conf).items()}

    num_points, num_images = _extract_reconstruction_stats(reconstruction_dir)
    eval_prediction_csv = metrics_dir / "eval_predictions.csv"
    eval_prediction_csv, registration_rate = _build_eval_prediction_csv(
        reconstruction_dir=reconstruction_dir,
        out_csv=eval_prediction_csv,
    )
    maa, per_dataset = _compute_maa(eval_prediction_csv)

    metrics = {
        "maa": float(maa),
        "num_points": int(num_points),
        "num_images": int(num_images),
        "registration_rate": float(registration_rate),
    }

    metrics_path = metrics_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    final_ply_path = _ensure_final_ply(reconstruction_dir, metrics_dir, num_points)

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

        mlflow.log_metric("maa", float(maa))
        mlflow.log_metric("num_points", int(num_points))
        mlflow.log_metric("num_images", int(num_images))
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

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
