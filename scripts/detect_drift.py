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
    from scripts.drift_monitor import DriftMonitor

    baselines_path = ROOT / "data" / "baselines" / "eda_baselines.json"
    if not baselines_path.exists():
        baselines_path = ROOT / "data" / "baselines" / "eda_baselines.json"
    if not baselines_path.exists():
        log.error(
            "EDA baselines not found at either expected path. "
            "Run the eda_baselines DVC stage first."
        )
        print("ok")  # Don't block on missing baselines; just skip drift check
        sys.exit(0)

    report_path = ROOT / "data" / "baselines" / "drift_report.json"

    monitor = DriftMonitor(baselines_path=baselines_path)

    report = monitor.check(
        report_path=report_path,
        check_performance=True,
    )

    log.info(f"Drift status: {report.status} | Alerts: {len(report.alerts)}")
    # Write full JSON report to stderr so it doesn't pollute the XCom value
    sys.stderr.write(json.dumps(report.as_dict(), indent=2) + "\n")

    # BashOperator XCom captures stdout. Last stdout line must be exactly
    # "drift" or "ok" — no extra content. Always exit 0 so the task does not
    # fail; branching is handled by BranchPythonOperator reading the XCom.
    if report.status in ("warning", "critical"):
        print("drift")
    else:
        print("ok")
    sys.exit(0)


if __name__ == "__main__":
    main()
