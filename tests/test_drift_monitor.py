import pytest
import numpy as np
import json
from scripts.drift_monitor import DriftMonitor

def test_drift_monitor_no_drift(tmp_path):
    baselines = {
        "n_images": {"mean": 100, "std": 10},
        "registration_rate": {"mean": 0.8, "std": 0.1, "p50": 0.82, "p25": 0.75},
        "inference_latency_seconds": {"mean": 60, "std": 10}
    }
    baseline_path = tmp_path / "baselines.json"
    baseline_path.write_text(json.dumps(baselines))
    
    monitor = DriftMonitor(baseline_path)
    
    np.random.seed(42)
    reference = np.random.normal(0, 1, 100)
    current = np.random.normal(0, 1, 100)
    
    has_drift, p_val = monitor._check_drift(reference, current, threshold=0.05)
    assert has_drift is False
    assert p_val > 0.05

def test_drift_monitor_performance_drift(tmp_path):
    baselines = {
        "n_images": {"mean": 100, "std": 10},
        "registration_rate": {"mean": 0.8, "std": 0.1, "p50": 0.80, "p25": 0.75},
        "inference_latency_seconds": {"mean": 60, "std": 10}
    }
    baseline_path = tmp_path / "baselines.json"
    baseline_path.write_text(json.dumps(baselines))
    
    monitor = DriftMonitor(baseline_path)
    
    alert = monitor._check_performance_drift(0.4)
    assert alert is not None
    assert alert["metric"] == "registration_rate"
    assert alert["severity"] == "critical"
    
    alert2 = monitor._check_performance_drift(0.78)
    assert alert2 is None
