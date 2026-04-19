"""
scripts/eda_baselines.py
─────────────────────────
Stage 1 — EDA Baselines.

Computes statistical baselines from the training dataset for drift detection:
  - Image resolution distribution (mean, std, min, max)
  - Sharpness scores (Laplacian variance per image)
  - Descriptor norm baselines (populated after feature extraction)
  - Scene/dataset inventory

Outputs:
  - data/baselines/eda_baselines.json
  - data/baselines/eda_metrics.json
  - data/baselines/similarity_matrix.png
  - data/baselines/sharpness_hist.png
  - data/baselines/resolution_hist.png
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import IMC2025TrainData, DEFAULT_DATASET_DIR


def compute_sharpness(image_path: str) -> float:
    """Compute Laplacian variance as a sharpness score."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def _log_to_mlflow(eda_metrics: dict, baselines_path: Path, output_dir: Path) -> None:
    """Best-effort MLflow logging for Stage-1 EDA outputs."""
    try:
        import mlflow

        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("scene_reconstruction_dvc")

        with mlflow.start_run(run_name="eda_baselines"):
            mlflow.log_metrics({k: float(v) for k, v in eda_metrics.items()})
            mlflow.log_artifact(str(baselines_path), artifact_path="baselines")
            mlflow.log_artifact(str(output_dir / "eda_metrics.json"), artifact_path="metrics")

            for plot_name in (
                "sharpness_hist.png",
                "resolution_hist.png",
                "similarity_matrix.png",
            ):
                plot_path = output_dir / plot_name
                if plot_path.exists():
                    mlflow.log_artifact(str(plot_path), artifact_path="plots")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[warn] MLflow logging skipped: {exc}")


def main() -> None:
    output_dir = Path(DEFAULT_DATASET_DIR) / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = IMC2025TrainData.create(DEFAULT_DATASET_DIR)
    schema.preprocess()
    df = schema.df

    # --- Image resolution stats ---
    widths, heights = [], []
    sharpness_scores = []
    sample_paths = []

    for _, row in df.iterrows():
        img_path = schema.resolve_image_path(row)
        sample_paths.append(img_path)

    # Sample up to 500 images for EDA
    sample_size = min(500, len(sample_paths))
    rng = np.random.default_rng(42)
    sampled = rng.choice(sample_paths, size=sample_size, replace=False)

    brightness_scores = []
    contrast_scores = []

    for img_path in sampled:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        brightness_scores.append(float(gray.mean()))
        contrast_scores.append(float(gray.std()))

    widths_arr = np.array(widths, dtype=float)
    heights_arr = np.array(heights, dtype=float)
    sharpness_arr = np.array(sharpness_scores, dtype=float)
    brightness_arr = np.array(brightness_scores, dtype=float)
    contrast_arr = np.array(contrast_scores, dtype=float)

    # --- Determine blurry threshold (10th percentile of sharpness) ---
    blurry_threshold = float(np.percentile(sharpness_arr, 10)) if len(sharpness_arr) > 0 else 100.0

    baselines = {
        "dataset": {
            "total_images": len(df),
            "total_scenes": df.groupby(["dataset", "scene"]).ngroups,
            "total_datasets": df["dataset"].nunique(),
            "sampled_for_eda": sample_size,
        },
        "resolution": {
            "width_mean": float(widths_arr.mean()) if len(widths_arr) else 0,
            "width_std": float(widths_arr.std()) if len(widths_arr) else 0,
            "height_mean": float(heights_arr.mean()) if len(heights_arr) else 0,
            "height_std": float(heights_arr.std()) if len(heights_arr) else 0,
        },
        "sharpness": {
            "mean": float(sharpness_arr.mean()) if len(sharpness_arr) else 0,
            "std": float(sharpness_arr.std()) if len(sharpness_arr) else 0,
            "p10": float(np.percentile(sharpness_arr, 10)) if len(sharpness_arr) else 0,
            "p90": float(np.percentile(sharpness_arr, 90)) if len(sharpness_arr) else 0,
            "blurry_threshold": blurry_threshold,
            "non_upright_pct": 0.0,  # populated by orientation check
        },
        "brightness": {
            "mean": float(brightness_arr.mean()) if len(brightness_arr) else 128.0,
            "std": float(brightness_arr.std()) if len(brightness_arr) else 1.0,
            "p10": float(np.percentile(brightness_arr, 10)) if len(brightness_arr) else 0,
            "p90": float(np.percentile(brightness_arr, 90)) if len(brightness_arr) else 255,
        },
        "contrast": {
            "mean": float(contrast_arr.mean()) if len(contrast_arr) else 50.0,
            "std": float(contrast_arr.std()) if len(contrast_arr) else 1.0,
            "p10": float(np.percentile(contrast_arr, 10)) if len(contrast_arr) else 0,
            "p90": float(np.percentile(contrast_arr, 90)) if len(contrast_arr) else 100,
        },
        "descriptor": {
            "norm_mean": 0.0,   # populated after feature extraction
            "norm_std": 0.0,
            "norm_p10": 0.0,
            "norm_p90": 0.0,
        },
        "orientation": {
            "non_upright_pct": 0.0,
        },
    }

    # Write baselines
    with open(output_dir / "eda_baselines.json", "w") as f:
        json.dump(baselines, f, indent=2)

    # Write EDA metrics (DVC-tracked)
    eda_metrics = {
        "total_images": baselines["dataset"]["total_images"],
        "total_scenes": baselines["dataset"]["total_scenes"],
        "sharpness_mean": baselines["sharpness"]["mean"],
        "sharpness_p10": baselines["sharpness"]["p10"],
        "resolution_width_mean": baselines["resolution"]["width_mean"],
        "resolution_height_mean": baselines["resolution"]["height_mean"],
        "blurry_threshold": blurry_threshold,
        "brightness_mean": baselines["brightness"]["mean"],
        "contrast_mean": baselines["contrast"]["mean"],
    }
    with open(output_dir / "eda_metrics.json", "w") as f:
        json.dump(eda_metrics, f, indent=2)

    # --- Generate histogram plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Sharpness histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(sharpness_arr, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(blurry_threshold, color="red", linestyle="--", label=f"Blurry threshold ({blurry_threshold:.0f})")
        ax.set_xlabel("Laplacian Variance (Sharpness)")
        ax.set_ylabel("Count")
        ax.set_title("Image Sharpness Distribution")
        ax.legend()
        fig.savefig(output_dir / "sharpness_hist.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Resolution histogram
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(widths_arr, bins=30, edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Width (px)")
        axes[0].set_title("Image Width Distribution")
        axes[1].hist(heights_arr, bins=30, edgecolor="black", alpha=0.7)
        axes[1].set_xlabel("Height (px)")
        axes[1].set_title("Image Height Distribution")
        fig.suptitle("Image Resolution Distribution")
        fig.savefig(output_dir / "resolution_hist.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Similarity matrix placeholder (needs descriptor extraction first)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "Run feature extraction first", ha="center", va="center", fontsize=14)
        ax.set_title("Descriptor Similarity Matrix")
        fig.savefig(output_dir / "similarity_matrix.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    except ImportError:
        print("[warn] matplotlib not available — skipping plot generation")

    _log_to_mlflow(
        eda_metrics=eda_metrics,
        baselines_path=output_dir / "eda_baselines.json",
        output_dir=output_dir,
    )

    print(json.dumps(baselines, indent=2))
    print(f"[OK] EDA baselines written to {output_dir / 'eda_baselines.json'}")


if __name__ == "__main__":
    main()
