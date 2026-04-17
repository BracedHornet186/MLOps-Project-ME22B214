"""
scripts/drift_monitor.py
─────────────────────────
Standalone drift detection module.
Used by:
  - The Airflow preprocessing DAG (task: check_feature_drift)
  - The Airflow retraining DAG (task: run_drift_check)
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

# Performance proxy thresholds
DEFAULT_REG_RATE_DROP_THRESHOLD = float(
    os.environ.get("DRIFT_REG_RATE_DROP", "0.20")
)  # alert if registration_rate drops >20% from baseline
DEFAULT_MAA_THRESHOLD = float(
    os.environ.get("DRIFT_MAA_THRESHOLD", "0.45")
)


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
      4. Brightness distribution (mean pixel intensity drift)
      5. Contrast distribution (pixel intensity std drift)
      6. Performance proxy (registration_rate / mAA from MLflow)
    """

    def __init__(
        self,
        baselines_path: Path = DEFAULT_BASELINES_PATH,
        features_dir: Path = DEFAULT_FEATURES_DIR,
        ks_alpha: float = DEFAULT_KS_ALPHA,
        mlflow_uri: Optional[str] = None,
        reg_rate_drop_threshold: float = DEFAULT_REG_RATE_DROP_THRESHOLD,
        maa_threshold: float = DEFAULT_MAA_THRESHOLD,
    ):
        self.features_dir = features_dir
        self.ks_alpha = ks_alpha
        self.baselines = self._load_baselines(baselines_path)
        self.mlflow_uri = mlflow_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.reg_rate_drop_threshold = reg_rate_drop_threshold
        self.maa_threshold = maa_threshold

    # ── Public API ─────────────────────────────────────────────────────────────

    def check(
        self,
        live_stats: Optional[dict] = None,
        report_path: Path = DEFAULT_REPORT_PATH,
        check_performance: bool = False,
    ) -> DriftReport:
        """
        Run all drift checks.

        Args:
            live_stats: Optional dict of live scalar statistics.
                        If None, reads from saved feature .npy files.
            report_path: Where to write the JSON report.
            check_performance: If True, query MLflow for performance proxy.

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

        # ── 4. Brightness drift ────────────────────────────────────────────
        if live_stats:
            alerts = self._check_brightness(live_stats)
            report.alerts.extend(alerts)
            report.n_features_checked += 1

        # ── 5. Contrast drift ──────────────────────────────────────────────
        if live_stats:
            alerts = self._check_contrast(live_stats)
            report.alerts.extend(alerts)
            report.n_features_checked += 1

        # ── 6. Performance proxy drift ─────────────────────────────────────
        if check_performance:
            alerts = self._check_performance_proxy()
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

    def _check_brightness(self, live_stats: dict) -> list[DriftAlert]:
        """Check if mean pixel brightness has drifted from baseline."""
        alerts: list[DriftAlert] = []
        baseline_bright = self.baselines.get("brightness", {})
        if not baseline_bright:
            return alerts

        b_mean = baseline_bright.get("mean", 128.0)
        b_std = baseline_bright.get("std", 1.0) or 1.0
        live_val = live_stats.get("brightness_mean")
        if live_val is None:
            return alerts

        z = abs(float(live_val) - b_mean) / b_std
        if z > 3.0:
            alerts.append(DriftAlert(
                feature="brightness_mean",
                stat_name="z_score",
                baseline_val=b_mean,
                live_val=float(live_val),
                severity="warning" if z < 5 else "critical",
                message=(
                    f"BRIGHTNESS DRIFT: live={live_val:.1f}, "
                    f"baseline={b_mean:.1f}, z={z:.2f}. "
                    "Images may be significantly brighter/darker than training data."
                ),
            ))
        return alerts

    def _check_contrast(self, live_stats: dict) -> list[DriftAlert]:
        """Check if image contrast (std of pixel intensity) has drifted."""
        alerts: list[DriftAlert] = []
        baseline_contrast = self.baselines.get("contrast", {})
        if not baseline_contrast:
            return alerts

        b_mean = baseline_contrast.get("mean", 50.0)
        b_std = baseline_contrast.get("std", 1.0) or 1.0
        live_val = live_stats.get("contrast_mean")
        if live_val is None:
            return alerts

        z = abs(float(live_val) - b_mean) / b_std
        if z > 3.0:
            alerts.append(DriftAlert(
                feature="contrast_mean",
                stat_name="z_score",
                baseline_val=b_mean,
                live_val=float(live_val),
                severity="warning" if z < 5 else "critical",
                message=(
                    f"CONTRAST DRIFT: live={live_val:.1f}, "
                    f"baseline={b_mean:.1f}, z={z:.2f}. "
                    "Images may have very different contrast than training data."
                ),
            ))
        return alerts

    def _check_performance_proxy(self) -> list[DriftAlert]:
        """
        Query MLflow for recent runs and check if registration_rate or mAA
        have dropped significantly from prior production baselines.
        """
        alerts: list[DriftAlert] = []
        try:
            import mlflow

            mlflow.set_tracking_uri(self.mlflow_uri)
            client = mlflow.tracking.MlflowClient(tracking_uri=self.mlflow_uri)

            # Find the scene_reconstruction experiment
            experiment = client.get_experiment_by_name("scene_reconstruction")
            if experiment is None:
                logger.info("No scene_reconstruction experiment found — skipping performance check")
                return alerts

            # Get the most recent finished runs
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                order_by=["attributes.start_time DESC"],
                max_results=5,
            )
            if not runs:
                return alerts

            latest_run = runs[0]
            latest_maa = latest_run.data.metrics.get("mAA_overall")
            latest_reg_rate = latest_run.data.metrics.get("registration_rate")

            # Check mAA threshold
            if latest_maa is not None and latest_maa < self.maa_threshold:
                severity = "critical" if latest_maa < (self.maa_threshold * 0.8) else "warning"
                alerts.append(DriftAlert(
                    feature="mAA_overall",
                    stat_name="threshold_check",
                    baseline_val=self.maa_threshold,
                    live_val=latest_maa,
                    severity=severity,
                    message=(
                        f"PERFORMANCE DECAY [mAA]: latest={latest_maa:.4f} "
                        f"< threshold={self.maa_threshold:.2f}. "
                        f"Model accuracy may have degraded."
                    ),
                ))

            # Check registration_rate drop across recent runs
            if latest_reg_rate is not None and len(runs) >= 2:
                # Compare against the average of previous runs
                prev_reg_rates = [
                    r.data.metrics.get("registration_rate", 0)
                    for r in runs[1:]
                    if r.data.metrics.get("registration_rate") is not None
                ]
                if prev_reg_rates:
                    avg_prev = sum(prev_reg_rates) / len(prev_reg_rates)
                    if avg_prev > 0:
                        drop = (avg_prev - latest_reg_rate) / avg_prev
                        if drop > self.reg_rate_drop_threshold:
                            alerts.append(DriftAlert(
                                feature="registration_rate",
                                stat_name="relative_drop",
                                baseline_val=avg_prev,
                                live_val=latest_reg_rate,
                                severity="critical" if drop > 0.4 else "warning",
                                message=(
                                    f"PERFORMANCE DECAY [registration_rate]: "
                                    f"latest={latest_reg_rate:.3f}, "
                                    f"avg_previous={avg_prev:.3f}, "
                                    f"drop={drop:.1%}. "
                                    f"Significantly fewer images are being registered."
                                ),
                            ))
        except ImportError:
            logger.warning("mlflow not available — skipping performance proxy check")
        except Exception as e:
            logger.warning(f"Performance proxy check failed: {e}")

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
    BRIGHTNESS_DRIFT_GAUGE = Gauge(
        "brightness_drift_z",
        "Z-score of brightness vs EDA baseline",
    )
    CONTRAST_DRIFT_GAUGE = Gauge(
        "contrast_drift_z",
        "Z-score of contrast vs EDA baseline",
    )
    PERFORMANCE_PROXY_GAUGE = Gauge(
        "performance_proxy_status",
        "Performance proxy drift: 0=ok, 1=warning, 2=critical",
    )

    def update_prometheus_drift_metrics(report: DriftReport) -> None:
        status_map = {"ok": 0, "warning": 1, "critical": 2}
        DRIFT_STATUS_GAUGE.set(status_map.get(report.status, 0))
        DRIFT_ALERT_COUNT_GAUGE.set(len(report.alerts))

        # Track per-feature gauges
        perf_severity = 0
        for alert in report.alerts:
            if alert.stat_name == "norm_z_score":
                model = alert.feature.replace("global_", "").replace("_norm_mean", "")
                DESCRIPTOR_NORM_DRIFT_GAUGE.labels(model=model).set(alert.live_val)
            elif alert.feature == "brightness_mean":
                z = abs(alert.live_val - alert.baseline_val) / max(
                    alert.baseline_val, 1
                )
                BRIGHTNESS_DRIFT_GAUGE.set(z)
            elif alert.feature == "contrast_mean":
                z = abs(alert.live_val - alert.baseline_val) / max(
                    alert.baseline_val, 1
                )
                CONTRAST_DRIFT_GAUGE.set(z)
            elif alert.feature in ("mAA_overall", "registration_rate"):
                sev = 2 if alert.severity == "critical" else 1
                perf_severity = max(perf_severity, sev)

        PERFORMANCE_PROXY_GAUGE.set(perf_severity)

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
    parser.add_argument("--check-performance", action="store_true",
                        help="Also check MLflow performance proxy drift")
    parser.add_argument("--mlflow-uri", default=None,
                        help="MLflow tracking URI for performance checks")
    args = parser.parse_args()

    monitor = DriftMonitor(
        baselines_path=Path(args.baselines),
        features_dir=Path(args.features_dir),
        ks_alpha=args.alpha,
        mlflow_uri=args.mlflow_uri,
    )
    report = monitor.check(
        report_path=Path(args.report_out),
        check_performance=args.check_performance,
    )
    print(json.dumps(report.as_dict(), indent=2))
    sys.exit(0 if report.status == "ok" else 1)


if __name__ == "__main__":
    main()

