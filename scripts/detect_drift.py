"""
scripts/detect_drift.py
───────────────────────
Lightweight drift detection wrapper for Airflow.

Compares current inference stats against EDA baselines.
Exit codes:
  0 → no drift detected
  1 → drift detected

Pushes "drift" or "ok" to stdout (last line) for Airflow xcom.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    # Import the existing drift monitor
    sys.path.insert(0, str(ROOT / "scripts"))
    from drift_monitor import DriftMonitor

    baselines_path = ROOT / "data" / "baselines" / "eda_baselines.json"
    if not baselines_path.exists():
        # Try the processed path as fallback
        baselines_path = ROOT / "data" / "processed" / "eda_baselines.json"

    features_dir = ROOT / "data" / "processed" / "features"
    report_path = ROOT / "data" / "processed" / "drift_report.json"

    monitor = DriftMonitor(
        baselines_path=baselines_path,
        features_dir=features_dir,
    )

    report = monitor.check(
        report_path=report_path,
        check_performance=True,
    )

    log.info(f"Drift status: {report.status} | Alerts: {len(report.alerts)}")
    print(json.dumps(report.as_dict(), indent=2))

    if report.status in ("warning", "critical"):
        # Push xcom value for Airflow branching
        print("drift")
        sys.exit(1)
    else:
        print("ok")
        sys.exit(0)


if __name__ == "__main__":
    main()
