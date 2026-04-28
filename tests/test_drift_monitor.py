import pytest
import numpy as np
import json
from scripts.drift_monitor import DriftMonitor

def test_drift_monitor_no_drift(tmp_path):
    baselines = {
        "brightness": {"mean": 128.0, "std": 10.0}
    }
    baseline_path = tmp_path / "baselines.json"
    baseline_path.write_text(json.dumps(baselines))
    
    monitor = DriftMonitor(baseline_path)
    
    live_stats = {"brightness_mean": 129.0}
    alerts = monitor._check_brightness(live_stats)
    
    assert len(alerts) == 0

def test_drift_monitor_brightness_drift(tmp_path):
    baselines = {
        "brightness": {"mean": 128.0, "std": 10.0}
    }
    baseline_path = tmp_path / "baselines.json"
    baseline_path.write_text(json.dumps(baselines))
    
    monitor = DriftMonitor(baseline_path)
    
    # 170 is 4.2 std deviations away, which triggers warning (>3.0)
    live_stats = {"brightness_mean": 170.0}
    alerts = monitor._check_brightness(live_stats)
    
    assert len(alerts) == 1
    assert alerts[0].feature == "brightness_mean"
    assert alerts[0].severity == "warning"
    
    # 200 is 7.2 std deviations away, which triggers critical (>5.0)
    live_stats_crit = {"brightness_mean": 200.0}
    alerts_crit = monitor._check_brightness(live_stats_crit)
    
    assert len(alerts_crit) == 1
    assert alerts_crit[0].severity == "critical"
