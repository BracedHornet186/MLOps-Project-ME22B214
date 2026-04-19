"""
scripts/scene_inventory.py
───────────────────────────
Stage 1 — Scene Inventory.

Generates a CSV summary of all scenes in the training dataset:
  - Dataset, scene, number of images, image paths

Outputs:
  - data/processed/scene_inventory.csv
  - data/processed/inventory_metrics.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import IMC2025TrainData, DEFAULT_DATASET_DIR


def _log_to_mlflow(metrics: dict, inventory_path: Path, metrics_path: Path) -> None:
    """Best-effort MLflow logging for scene inventory metrics/artifacts."""
    try:
        import mlflow

        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("stage1_scene_inventory")

        with mlflow.start_run(run_name="scene_inventory"):
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            mlflow.log_artifact(str(inventory_path), artifact_path="inventory")
            mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[warn] MLflow logging skipped: {exc}")


def main() -> None:
    output_dir = Path(DEFAULT_DATASET_DIR) / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = IMC2025TrainData.create(DEFAULT_DATASET_DIR)
    schema.preprocess()
    df = schema.df

    # Build scene inventory
    inventory = (
        df.groupby(["dataset", "scene"])
        .agg(
            n_images=("image", "count"),
            images=("image", list),
        )
        .reset_index()
    )

    # Save inventory CSV (without the full image lists for readability)
    inventory_csv = inventory[["dataset", "scene", "n_images"]].copy()
    inventory_csv.to_csv(output_dir / "scene_inventory.csv", index=False)

    # Metrics
    metrics = {
        "total_datasets": int(df["dataset"].nunique()),
        "total_scenes": int(inventory.shape[0]),
        "total_images": int(df.shape[0]),
        "min_images_per_scene": int(inventory["n_images"].min()),
        "max_images_per_scene": int(inventory["n_images"].max()),
        "mean_images_per_scene": round(float(inventory["n_images"].mean()), 1),
        "scenes_with_lt_5_images": int((inventory["n_images"] < 5).sum()),
    }

    with open(output_dir / "inventory_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    _log_to_mlflow(
        metrics=metrics,
        inventory_path=output_dir / "scene_inventory.csv",
        metrics_path=output_dir / "inventory_metrics.json",
    )

    print(json.dumps(metrics, indent=2))
    print(f"[OK] Scene inventory saved — {metrics['total_scenes']} scenes across {metrics['total_datasets']} datasets")


if __name__ == "__main__":
    main()
