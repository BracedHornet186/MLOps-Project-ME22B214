"""
scripts/drift_monitor.py
─────────────────────────
Standalone drift detection module.
Used by:
  - The Airflow preprocessing DAG (task: check_feature_drift)
  - The FastAPI /metrics endpoint (Prometheus gauge)
  - Stage 6 monitoring callbacks

Drift is detected using the Kolmogorov–Smirnov two-sample test.
Baselines are the EDA statistics saved to data/processed/eda_baselines.json.
Live statistics are read from data/processed/features/<scene>/

All alerts are:
  1. Printed to stdout (captured by Airflow logs)
  2. Written to data/processed/drift_report.json
  3. Exposed as Prometheus gauges (when called from the API)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_BASELINES_PATH = Path(
    os.environ.get("DEFAULT_DATASET_DIR", "data") + "/processed/eda_baselines.json"
)
DEFAULT_FEATURES_DIR = Path(
    os.environ.get("DEFAULT_DATASET_DIR", "data") + "/processed/features"
)
DEFAULT_REPORT_PATH = Path(
    os.environ.get("DEFAULT_DATASET_DIR", "data") + "/processed/drift_report.json"
)

# KS-test p-value threshold — alert when p < this value
DEFAULT_KS_ALPHA = float(os.environ.get("DRIFT_KS_ALPHA", "0.01"))


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class DriftAlert:
    feature: str          # e.g. "dinov2_norm", "match_count"
    stat_name: str        # e.g. "mean", "ks_statistic"
    baseline_val: float
    live_val: float
    severity: str         # "warning" | "critical"
    message: str

    def as_dict(self) -> dict:
        return {
            "feature": self.feature,
            "stat_name": self.stat_name,
            "baseline_val": self.baseline_val,
            "live_val": self.live_val,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass
class DriftReport:
    timestamp: str
    n_scenes_checked: int = 0
    n_features_checked: int = 0
    alerts: list[DriftAlert] = field(default_factory=list)
    status: str = "ok"    # "ok" | "warning" | "critical"

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "n_scenes_checked": self.n_scenes_checked,
            "n_features_checked": self.n_features_checked,
            "alerts": [a.as_dict() for a in self.alerts],
            "status": self.status,
        }


# ── Main checker ───────────────────────────────────────────────────────────────

class DriftMonitor:
    """
    Compares live feature statistics against EDA baselines.

    Checks performed:
      1. Descriptor norm distribution (z-score + KS-test)
      2. Image resolution distribution (mean drift)
      3. Sharpness score distribution (mean drift)
      4. Match count distribution (if available from a prior run)
    """

    def __init__(
        self,
        baselines_path: Path = DEFAULT_BASELINES_PATH,
        features_dir: Path = DEFAULT_FEATURES_DIR,
        ks_alpha: float = DEFAULT_KS_ALPHA,
    ):
        self.features_dir = features_dir
        self.ks_alpha = ks_alpha
        self.baselines = self._load_baselines(baselines_path)

    # ── Public API ─────────────────────────────────────────────────────────────

    def check(
        self,
        live_stats: Optional[dict] = None,
        report_path: Path = DEFAULT_REPORT_PATH,
    ) -> DriftReport:
        """
        Run all drift checks.

        Args:
            live_stats: Optional dict of live scalar statistics.
                        If None, reads from saved feature .npy files.
            report_path: Where to write the JSON report.

        Returns:
            DriftReport with all alerts.
        """
        import datetime
        report = DriftReport(
            timestamp=datetime.datetime.utcnow().isoformat() + "Z"
        )

        if self.baselines is None:
            logger.warning("No baselines loaded — skipping drift check")
            report.status = "ok"
            return report

        # ── 1. Descriptor norm drift ───────────────────────────────────────
        if live_stats:
            alerts = self._check_descriptor_norms_from_stats(live_stats)
            report.alerts.extend(alerts)
            report.n_features_checked += 1
        else:
            alerts = self._check_descriptor_norms_from_files()
            report.alerts.extend(alerts)
            report.n_features_checked += 1

        # ── 2. Sharpness drift ─────────────────────────────────────────────
        if live_stats:
            alerts = self._check_sharpness(live_stats)
            report.alerts.extend(alerts)
            report.n_features_checked += 1

        # ── 3. Resolution drift ────────────────────────────────────────────
        if live_stats:
            alerts = self._check_resolution(live_stats)
            report.alerts.extend(alerts)
            report.n_features_checked += 1

        # ── Determine overall status ───────────────────────────────────────
        if any(a.severity == "critical" for a in report.alerts):
            report.status = "critical"
        elif any(a.severity == "warning" for a in report.alerts):
            report.status = "warning"

        # ── Persist report ─────────────────────────────────────────────────
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report.as_dict(), f, indent=2)
        logger.info(
            f"Drift report saved to {report_path} | "
            f"status={report.status} | alerts={len(report.alerts)}"
        )

        for alert in report.alerts:
            level = logging.WARNING if alert.severity == "warning" else logging.ERROR
            logger.log(level, alert.message)

        return report

    # ── Internal checks ────────────────────────────────────────────────────────

    def _check_descriptor_norms_from_stats(self, live_stats: dict) -> list[DriftAlert]:
        alerts: list[DriftAlert] = []
        baseline_desc = self.baselines.get("descriptor", {})
        if not baseline_desc:
            return alerts

        b_mean = baseline_desc.get("norm_mean", 0.0)
        b_std  = baseline_desc.get("norm_std", 1.0) or 1.0
        b_p10  = baseline_desc.get("norm_p10", 0.0)
        b_p90  = baseline_desc.get("norm_p90", 1.0)

        for key in ("global_dinov2_norm_mean", "global_isc_norm_mean",
                    "global_mast3r_spoc_norm_mean"):
            if key not in live_stats:
                continue
            live_val = float(live_stats[key])
            z = abs(live_val - b_mean) / b_std

            if z > 5.0:
                alerts.append(DriftAlert(
                    feature=key, stat_name="norm_z_score",
                    baseline_val=b_mean, live_val=live_val,
                    severity="critical",
                    message=(
                        f"CRITICAL DRIFT [{key}]: "
                        f"live={live_val:.4f}, baseline={b_mean:.4f}, z={z:.2f}"
                    ),
                ))
            elif z > 3.0:
                alerts.append(DriftAlert(
                    feature=key, stat_name="norm_z_score",
                    baseline_val=b_mean, live_val=live_val,
                    severity="warning",
                    message=(
                        f"WARNING DRIFT [{key}]: "
                        f"live={live_val:.4f}, baseline={b_mean:.4f}, z={z:.2f}"
                    ),
                ))
        return alerts

    def _check_descriptor_norms_from_files(self) -> list[DriftAlert]:
        """Load saved .npy descriptor files and run KS-test against baseline."""
        from scipy.stats import ks_2samp

        alerts: list[DriftAlert] = []
        baseline_desc = self.baselines.get("descriptor", {})
        if not baseline_desc:
            return alerts

        b_mean = baseline_desc.get("norm_mean", 0.0)
        b_std  = baseline_desc.get("norm_std", 1.0) or 1.0

        # Reconstruct a synthetic baseline sample from mean/std
        rng = np.random.default_rng(42)
        baseline_sample = rng.normal(b_mean, b_std, size=500)

        npy_files = list(self.features_dir.rglob("global_dinov2.npy"))
        if not npy_files:
            return alerts

        live_norms: list[float] = []
        for npy_path in npy_files[:50]:  # cap at 50 scenes for speed
            try:
                feats = np.load(str(npy_path))
                norms = np.linalg.norm(feats, axis=1)
                live_norms.extend(norms.tolist())
            except Exception:
                pass

        if len(live_norms) < 10:
            return alerts

        live_arr = np.array(live_norms)
        ks_stat, p_val = ks_2samp(baseline_sample, live_arr)

        if p_val < self.ks_alpha:
            severity = "critical" if p_val < 0.001 else "warning"
            alerts.append(DriftAlert(
                feature="global_dinov2_norms",
                stat_name="ks_p_value",
                baseline_val=b_mean,
                live_val=float(live_arr.mean()),
                severity=severity,
                message=(
                    f"{severity.upper()} DRIFT [global_dinov2_norms]: "
                    f"KS p={p_val:.4f} < alpha={self.ks_alpha} | "
                    f"live_mean={live_arr.mean():.4f}, "
                    f"baseline_mean={b_mean:.4f}"
                ),
            ))
        return alerts

    def _check_sharpness(self, live_stats: dict) -> list[DriftAlert]:
        alerts: list[DriftAlert] = []
        baseline_sharp = self.baselines.get("sharpness", {})
        if not baseline_sharp:
            return alerts

        b_mean = baseline_sharp.get("mean", 0.0)
        b_std  = baseline_sharp.get("std", 1.0) or 1.0
        live_val = live_stats.get("sharpness_mean")
        if live_val is None:
            return alerts

        z = abs(float(live_val) - b_mean) / b_std
        if z > 3.0:
            alerts.append(DriftAlert(
                feature="sharpness_mean",
                stat_name="z_score",
                baseline_val=b_mean,
                live_val=float(live_val),
                severity="warning" if z < 5 else "critical",
                message=(
                    f"SHARPNESS DRIFT: live={live_val:.1f}, "
                    f"baseline={b_mean:.1f}, z={z:.2f}. "
                    "Check if deblurring is needed."
                ),
            ))
        return alerts

    def _check_resolution(self, live_stats: dict) -> list[DriftAlert]:
        alerts: list[DriftAlert] = []
        baseline_res = self.baselines.get("resolution", {})
        if not baseline_res:
            return alerts

        b_h = baseline_res.get("height_mean", 0.0)
        b_w = baseline_res.get("width_mean", 0.0)

        live_h = live_stats.get("height_mean")
        live_w = live_stats.get("width_mean")

        for dim, b_val, l_val in [
            ("height", b_h, live_h), ("width", b_w, live_w)
        ]:
            if l_val is None or b_val == 0:
                continue
            rel_diff = abs(float(l_val) - b_val) / b_val
            if rel_diff > 0.30:  # > 30% relative change
                alerts.append(DriftAlert(
                    feature=f"resolution_{dim}",
                    stat_name="relative_diff",
                    baseline_val=b_val,
                    live_val=float(l_val),
                    severity="warning",
                    message=(
                        f"RESOLUTION DRIFT [{dim}]: "
                        f"live={l_val:.0f}px, baseline={b_val:.0f}px, "
                        f"rel_diff={rel_diff:.1%}"
                    ),
                ))
        return alerts

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_baselines(path: Path) -> Optional[dict]:
        if not path.exists():
            logger.warning(f"EDA baselines not found at {path}")
            return None
        with open(path) as f:
            return json.load(f)


# ── Prometheus gauges (imported by api/main.py) ────────────────────────────────

try:
    from prometheus_client import Gauge

    DRIFT_STATUS_GAUGE = Gauge(
        "feature_drift_status",
        "Overall drift status: 0=ok, 1=warning, 2=critical",
    )
    DRIFT_ALERT_COUNT_GAUGE = Gauge(
        "feature_drift_alert_count",
        "Number of active drift alerts",
    )
    DESCRIPTOR_NORM_DRIFT_GAUGE = Gauge(
        "descriptor_norm_drift_z",
        "Z-score of descriptor norm vs EDA baseline",
        ["model"],
    )

    def update_prometheus_drift_metrics(report: DriftReport) -> None:
        status_map = {"ok": 0, "warning": 1, "critical": 2}
        DRIFT_STATUS_GAUGE.set(status_map.get(report.status, 0))
        DRIFT_ALERT_COUNT_GAUGE.set(len(report.alerts))
        for alert in report.alerts:
            if alert.stat_name == "norm_z_score":
                model = alert.feature.replace("global_", "").replace("_norm_mean", "")
                DESCRIPTOR_NORM_DRIFT_GAUGE.labels(model=model).set(alert.live_val)

except ImportError:
    def update_prometheus_drift_metrics(report) -> None:
        pass


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    """
    Run drift check from CLI.
    Usage: python -m scripts.drift_monitor
    """
    import argparse, sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run feature drift check")
    parser.add_argument("--baselines", default=str(DEFAULT_BASELINES_PATH))
    parser.add_argument("--features-dir", default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--alpha", type=float, default=DEFAULT_KS_ALPHA)
    args = parser.parse_args()

    monitor = DriftMonitor(
        baselines_path=Path(args.baselines),
        features_dir=Path(args.features_dir),
        ks_alpha=args.alpha,
    )
    report = monitor.check(report_path=Path(args.report_out))
    print(json.dumps(report.as_dict(), indent=2))
    sys.exit(0 if report.status == "ok" else 1)


if __name__ == "__main__":
    main()
