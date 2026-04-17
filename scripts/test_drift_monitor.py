"""
scripts/test_drift_monitor.py
─────────────────────────────
Unit tests for drift_monitor.py — covers the new brightness, contrast,
and performance proxy checks alongside the existing ones.

Run: python -m pytest scripts/test_drift_monitor.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.drift_monitor import DriftMonitor, DriftReport


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_baselines(**overrides) -> dict:
    """Build a minimal EDA baselines dict with sensible defaults."""
    baselines = {
        "descriptor": {
            "norm_mean": 10.0,
            "norm_std": 2.0,
            "norm_p10": 6.0,
            "norm_p90": 14.0,
        },
        "sharpness": {"mean": 500.0, "std": 100.0},
        "resolution": {
            "height_mean": 1080.0,
            "width_mean": 1920.0,
        },
        "brightness": {"mean": 120.0, "std": 15.0},
        "contrast": {"mean": 55.0, "std": 8.0},
    }
    baselines.update(overrides)
    return baselines


def _write_baselines(tmp_dir: Path, baselines: dict) -> Path:
    path = tmp_dir / "eda_baselines.json"
    path.write_text(json.dumps(baselines))
    return path


# ── Test brightness drift ────────────────────────────────────────────────────

class TestBrightnessDrift:
    def test_no_alert_when_within_range(self, tmp_path):
        bl = _write_baselines(tmp_path, _make_baselines())
        mon = DriftMonitor(baselines_path=bl, features_dir=tmp_path)
        report = mon.check(
            live_stats={"brightness_mean": 125.0},
            report_path=tmp_path / "report.json",
        )
        brightness_alerts = [a for a in report.alerts if a.feature == "brightness_mean"]
        assert len(brightness_alerts) == 0

    def test_warning_on_moderate_drift(self, tmp_path):
        bl = _write_baselines(tmp_path, _make_baselines())
        mon = DriftMonitor(baselines_path=bl, features_dir=tmp_path)
        # z = |50 - 120| / 15 = 4.67 → warning (3 < z < 5)
        report = mon.check(
            live_stats={"brightness_mean": 50.0},
            report_path=tmp_path / "report.json",
        )
        brightness_alerts = [a for a in report.alerts if a.feature == "brightness_mean"]
        assert len(brightness_alerts) == 1
        assert brightness_alerts[0].severity == "warning"

    def test_critical_on_extreme_drift(self, tmp_path):
        bl = _write_baselines(tmp_path, _make_baselines())
        mon = DriftMonitor(baselines_path=bl, features_dir=tmp_path)
        # z = |10 - 120| / 15 = 7.33 → critical (z >= 5)
        report = mon.check(
            live_stats={"brightness_mean": 10.0},
            report_path=tmp_path / "report.json",
        )
        brightness_alerts = [a for a in report.alerts if a.feature == "brightness_mean"]
        assert len(brightness_alerts) == 1
        assert brightness_alerts[0].severity == "critical"


# ── Test contrast drift ──────────────────────────────────────────────────────

class TestContrastDrift:
    def test_no_alert_when_within_range(self, tmp_path):
        bl = _write_baselines(tmp_path, _make_baselines())
        mon = DriftMonitor(baselines_path=bl, features_dir=tmp_path)
        report = mon.check(
            live_stats={"contrast_mean": 58.0},
            report_path=tmp_path / "report.json",
        )
        contrast_alerts = [a for a in report.alerts if a.feature == "contrast_mean"]
        assert len(contrast_alerts) == 0

    def test_warning_on_moderate_drift(self, tmp_path):
        bl = _write_baselines(tmp_path, _make_baselines())
        mon = DriftMonitor(baselines_path=bl, features_dir=tmp_path)
        # z = |25 - 55| / 8 = 3.75 → warning
        report = mon.check(
            live_stats={"contrast_mean": 25.0},
            report_path=tmp_path / "report.json",
        )
        contrast_alerts = [a for a in report.alerts if a.feature == "contrast_mean"]
        assert len(contrast_alerts) == 1
        assert contrast_alerts[0].severity == "warning"


# ── Test performance proxy drift ─────────────────────────────────────────────

class TestPerformanceProxy:
    @patch("scripts.drift_monitor.DriftMonitor._check_performance_proxy")
    def test_maa_below_threshold_triggers_alert(self, mock_perf, tmp_path):
        """Simulate a performance decay alert."""
        mock_perf.return_value = [
            MagicMock(
                feature="mAA_overall",
                stat_name="threshold_check",
                baseline_val=0.45,
                live_val=0.30,
                severity="critical",
                message="PERFORMANCE DECAY",
                as_dict=lambda: {
                    "feature": "mAA_overall",
                    "stat_name": "threshold_check",
                    "baseline_val": 0.45,
                    "live_val": 0.30,
                    "severity": "critical",
                    "message": "PERFORMANCE DECAY",
                },
            ),
        ]
        bl = _write_baselines(tmp_path, _make_baselines())
        mon = DriftMonitor(baselines_path=bl, features_dir=tmp_path)
        report = mon.check(
            report_path=tmp_path / "report.json",
            check_performance=True,
        )
        perf_alerts = [a for a in report.alerts if a.feature == "mAA_overall"]
        assert len(perf_alerts) == 1
        assert report.status == "critical"


# ── Test overall report status ───────────────────────────────────────────────

class TestReportStatus:
    def test_ok_when_no_drift(self, tmp_path):
        bl = _write_baselines(tmp_path, _make_baselines())
        mon = DriftMonitor(baselines_path=bl, features_dir=tmp_path)
        report = mon.check(
            live_stats={"brightness_mean": 120.0, "contrast_mean": 55.0},
            report_path=tmp_path / "report.json",
        )
        assert report.status == "ok"

    def test_report_persisted_to_disk(self, tmp_path):
        bl = _write_baselines(tmp_path, _make_baselines())
        mon = DriftMonitor(baselines_path=bl, features_dir=tmp_path)
        report_path = tmp_path / "report.json"
        mon.check(
            live_stats={"brightness_mean": 10.0},
            report_path=report_path,
        )
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert data["status"] in ("ok", "warning", "critical")
        assert "alerts" in data

    def test_no_baselines_returns_ok(self, tmp_path):
        mon = DriftMonitor(
            baselines_path=tmp_path / "nonexistent.json",
            features_dir=tmp_path,
        )
        report = mon.check(report_path=tmp_path / "report.json")
        assert report.status == "ok"
