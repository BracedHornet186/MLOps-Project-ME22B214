"""
scripts/prepare_submission.py
────────────────────────────────────────────────────────────────────────────
DVC Stage: prepare

Builds a submission-format DataFrame from preprocessed images so
that reconstruct_scenes.py can feed it directly into IMC2025Pipeline.run().

Reads:
  - data/train_labels.csv        (ground-truth structure)
  - data/processed/images/       (preprocessed image tree)

Writes:
  - data/prepared/prepared_input.csv

MLflow:
  - Child run "prepare" under parent DVC run (via MLFLOW_PARENT_RUN_ID)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare submission input for reconstruction")
    parser.add_argument(
        "--train-labels",
        default="data/train_labels.csv",
        help="Path to train_labels.csv",
    )
    parser.add_argument(
        "--processed-images-dir",
        default="data/processed/images",
        help="Directory containing preprocessed images (output of image_preprocess)",
    )
    parser.add_argument(
        "--output",
        default="data/prepared/prepared_input.csv",
        help="Output CSV path",
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


def _discover_processed_images(processed_dir: Path) -> set[str]:
    """Build a set of relative paths (from processed_dir) for all images found."""
    found = set()
    for p in sorted(processed_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            found.add(str(p.relative_to(processed_dir)))
    return found


def main() -> None:
    args = parse_args()

    train_labels_path = Path(args.train_labels)
    processed_dir = Path(args.processed_images_dir)
    output_path = Path(args.output)

    if not train_labels_path.exists():
        raise FileNotFoundError(f"Train labels not found: {train_labels_path}")
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed images dir not found: {processed_dir}")

    # ── Read ground-truth structure ──────────────────────────────────────
    gt_df = pd.read_csv(train_labels_path)
    log.info("Loaded %d rows from %s", len(gt_df), train_labels_path)

    # Discover available preprocessed images
    available_images = _discover_processed_images(processed_dir)
    log.info("Found %d preprocessed images under %s", len(available_images), processed_dir)

    # ── Build prepared DataFrame ─────────────────────────────────────────
    # IMC2025TrainData expects: dataset, scene, image, rotation_matrix, translation_vector
    # The data_root_dir will be data/processed/images, and
    # build_image_relative_path returns: train/{dataset}/{image}
    #
    # NaN poses are fine — the pipeline overwrites them with actual poses.
    # We use empty-string sentinel "nan;nan;..." so downstream CSV parsing
    # never encounters Python None or empty cells.
    nan_r = ";".join(["nan"] * 9)
    nan_t = ";".join(["nan"] * 3)

    rows = []
    matched = 0
    for _, row in gt_df.iterrows():
        dataset = str(row["dataset"])
        scene = str(row["scene"])
        image = str(row["image"])

        # The preprocessed path follows: train/{dataset}/{scene}_{image_name}
        # but the actual structure may be: train/{dataset}/{image}
        # Check which pattern exists
        rel_path = f"train/{dataset}/{image}"

        if rel_path in available_images:
            matched += 1

        rows.append({
            "dataset": dataset,
            "scene": scene,
            "image": image,
            "rotation_matrix": nan_r,
            "translation_vector": nan_t,
        })

    prepared_df = pd.DataFrame(rows)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_df.to_csv(output_path, index=False)

    log.info(
        "Prepared %d rows (%d with available images) → %s",
        len(prepared_df),
        matched,
        output_path,
    )

    # ── MLflow child run ─────────────────────────────────────────────────
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")
    run_tags = {"mlflow.parentRunId": parent_run_id} if parent_run_id else None

    with mlflow.start_run(run_name="prepare", nested=True, tags=run_tags):
        mlflow.log_param("train_labels", str(train_labels_path))
        mlflow.log_param("processed_images_dir", str(processed_dir))
        mlflow.log_param("total_rows", len(prepared_df))
        mlflow.log_param("matched_images", matched)

        mlflow.log_metric("total_rows", len(prepared_df))
        mlflow.log_metric("matched_images", matched)
        mlflow.log_metric("match_rate", float(matched / max(len(prepared_df), 1)))

        n_datasets = prepared_df["dataset"].nunique()
        n_scenes = prepared_df.groupby(["dataset", "scene"]).ngroups
        mlflow.log_metric("n_datasets", n_datasets)
        mlflow.log_metric("n_scenes", n_scenes)

        mlflow.log_artifact(str(output_path), artifact_path="prepared")

    log.info("MLflow child run 'prepare' logged successfully")


if __name__ == "__main__":
    main()
