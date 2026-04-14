"""
scripts/validate_data.py
─────────────────────────
Stage 1 — Data Validation.

Reads DVC-tracked CSVs, runs schema checks, writes:
  - data/processed/validation_report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from data import IMC2025TrainData, DEFAULT_DATASET_DIR


def main() -> None:
    output_dir = Path(DEFAULT_DATASET_DIR) / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = IMC2025TrainData.create(DEFAULT_DATASET_DIR)
    schema.preprocess()
    df = schema.df

    report = {
        "status": "ok",
        "total_images": len(df),
        "total_scenes": df.groupby(["dataset", "scene"]).ngroups,
        "missing_files": 0,        # dropped by preprocess()
        "duplicate_images": int(df.duplicated("image").sum()),
        "malformed_R": int(
            df["rotation_matrix"]
            .apply(lambda s: len(s.split(";")) != 9)
            .sum()
        ),
        "malformed_t": int(
            df["translation_vector"]
            .apply(lambda s: len(s.split(";")) != 3)
            .sum()
        ),
    }

    if any(
        v > 0
        for k, v in report.items()
        if k not in ("status", "total_images", "total_scenes")
    ):
        report["status"] = "warn"

    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    sys.exit(0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    main()