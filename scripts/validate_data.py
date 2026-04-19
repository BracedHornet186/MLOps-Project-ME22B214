"""
scripts/validate_data.py
─────────────────────────
Stage 1 — Data Validation.

Reads DVC-tracked CSVs, runs schema checks, writes:
    - data/baselines/validation_report.json
    - data/baselines/validation_metrics.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from data import IMC2025TrainData, DEFAULT_DATASET_DIR


def _count_malformed(series, expected_len: int) -> int:
    """Count rows whose semicolon-delimited vector has wrong length."""

    def _is_malformed(value) -> bool:
        parts = str(value).split(";")
        return len(parts) != expected_len

    return int(series.apply(_is_malformed).sum())


def _log_to_mlflow(metrics: dict, report_path: Path, metrics_path: Path, status: str) -> None:
    """Best-effort MLflow logging; never fails local/DVC execution."""
    try:
        import mlflow

        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("scene_reconstruction_dvc")

        with mlflow.start_run(run_name="validate_data"):
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(report_path), artifact_path="reports")
            mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
            mlflow.set_tag("validation_status", status)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[warn] MLflow logging skipped: {exc}")


def main() -> None:
    output_dir = Path(DEFAULT_DATASET_DIR) / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep missing-file counts before rows are dropped.
    schema = IMC2025TrainData.create(DEFAULT_DATASET_DIR)
    total_rows_before_drop = len(schema.df)
    schema.check_files_exist(drop=True, verbose=False)
    df = schema.df
    missing_files = max(total_rows_before_drop - len(df), 0)

    duplicate_images = int(
        df.duplicated(["dataset", "scene", "image"]).sum()
        if {"dataset", "scene", "image"}.issubset(df.columns)
        else df.duplicated().sum()
    )
    malformed_r = _count_malformed(df["rotation_matrix"], expected_len=9)
    malformed_t = _count_malformed(df["translation_vector"], expected_len=3)

    issue_count = int(missing_files + duplicate_images + malformed_r + malformed_t)
    status = "ok" if issue_count == 0 else "warn"

    report = {
        "status": status,
        "status_code": 0 if status == "ok" else 1,
        "total_rows_before_drop": int(total_rows_before_drop),
        "total_images": len(df),
        "total_scenes": df.groupby(["dataset", "scene"]).ngroups,
        "missing_files": missing_files,
        "duplicate_images": duplicate_images,
        "malformed_R": malformed_r,
        "malformed_t": malformed_t,
        "issue_count": issue_count,
    }

    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    metrics = {
        "total_rows_before_drop": int(total_rows_before_drop),
        "total_images": int(report["total_images"]),
        "total_scenes": int(report["total_scenes"]),
        "missing_files": int(missing_files),
        "duplicate_images": int(duplicate_images),
        "malformed_R": int(malformed_r),
        "malformed_t": int(malformed_t),
        "issue_count": int(issue_count),
        "status_code": int(report["status_code"]),
    }
    metrics_path = output_dir / "validation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    _log_to_mlflow(metrics, report_path, metrics_path, status=report["status"])

    print(json.dumps(report, indent=2))
    # Soft validation policy: keep full DVC pipeline reproducible even on warnings.
    sys.exit(0)


if __name__ == "__main__":
    main()