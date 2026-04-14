"""
dags/preprocess_features_dag.py
────────────────────────────────
Airflow DAG for Stage 2: Data Preprocessing & Feature Engineering.

DAG graph  (dvc dag equivalent):
  validate_data
       │
       ▼
  image_preprocess          ← orientation, deblur, segmentation, depth
       │
       ├──────────────┐
       ▼              ▼
  extract_global   extract_local
  _descriptors     _features
       │              │
       └──────┬───────┘
              ▼
       check_feature_drift
              │
              ▼
       update_dvc_metrics

Schedule: daily at 02:00 UTC.
Manual trigger: airflow dags trigger preprocess_features
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# ── Path bootstrap (same PYTHONPATH as set in docker-compose.yaml) ────────────
PROJECT_ROOT = Path(os.environ.get("PYTHONPATH", "/opt/airflow/project").split(":")[0])
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path(os.environ.get("DEFAULT_DATASET_DIR", f"{PROJECT_ROOT}/data"))
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

logger = logging.getLogger(__name__)

# ── Default DAG args ──────────────────────────────────────────────────────────

default_args = {
    "owner": "scene_reconstruction",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": __import__("datetime").timedelta(minutes=5),
    "email_on_failure": False,
}

# ── Task functions ─────────────────────────────────────────────────────────────

def task_validate_data(**ctx) -> None:
    """
    Task 1: Validate raw data using existing validate_data.py script.
    Reads DVC-tracked CSVs, runs schema checks, writes validation_report.json.
    """
    import subprocess, sys as _sys
    script = Path(PROJECT_ROOT) / "scripts" / "validate_data.py"
    result = subprocess.run(
        [_sys.executable, str(script)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"validate_data.py failed:\n{result.stderr}"
        )
    logger.info("Data validation passed")


def task_image_preprocess(**ctx) -> None:
    """
    Task 2: Run image-level preprocessing (orientation, deblur, segmentation, depth).
    Processes a sample of scenes for Airflow validation; full processing runs
    during pipeline inference via IMC2025Pipeline.run_scene().
    """
    import mlflow
    import torch
    from scripts.data import IMC2025TrainData
    from preprocesses.pipeline import PreprocessingPipeline, PreprocessingConfig
    from preprocesses.config import OrientationNormalizationConfig, DeblurringConfig
    from models.config import CheckOrientationModelConfig

    mlflow.set_tracking_uri(MLFLOW_URI)

    # Load baselines to get the correct blurry_threshold
    baselines_path = DATA_DIR / "processed" / "eda_baselines.json"
    blurry_threshold = 100.0
    if baselines_path.exists():
        with open(baselines_path) as f:
            baselines = json.load(f)
        blurry_threshold = baselines.get("sharpness", {}).get("blurry_threshold", 100.0)
    logger.info(f"Using blurry_threshold={blurry_threshold:.1f} from EDA baselines")

    # Build config — only run orientation + deblur in the DAG
    # (segmentation and depth only run per-inference, not as batch pre-steps)
    conf = PreprocessingConfig(
        orientation=OrientationNormalizationConfig(
            type="check_orientation",
            check_orientation=CheckOrientationModelConfig(
                weight_path="CHECK_ORIENTATION"
            ),
        ),
        deblurring=DeblurringConfig(
            type="fftformer",
            blurry_threshold=blurry_threshold,
        ),
        log_to_mlflow=True,
        mlflow_run_name="airflow_preprocess",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = PreprocessingPipeline(conf, device=device)

    # Process a sample of scenes (5 scenes max in the DAG to keep it fast)
    from pipelines.scene import Scene
    from scripts.data_schema import DataSchema
    data_schema = IMC2025TrainData.create(DATA_DIR)
    data_schema.preprocess()

    scenes_checked = 0
    for (dataset, scene_name), group in data_schema.df.groupby(["dataset", "scene"]):
        if scenes_checked >= 5:
            break
        image_paths = [
            data_schema.resolve_image_path(row)
            for _, row in group.iterrows()
        ]
        image_paths = [p for p in image_paths if Path(p).exists()]
        if not image_paths:
            continue

        scene = Scene(
            dataset=dataset,
            scene=scene_name,
            image_paths=image_paths,
            image_dir=str(Path(image_paths[0]).parent),
            data_schema=data_schema,
        )
        pipeline.run(scene)
        scenes_checked += 1

    logger.info(f"Image preprocessing complete: {scenes_checked} scenes processed")

    # Push blurry_threshold to XCom for downstream tasks
    ctx["ti"].xcom_push(key="blurry_threshold", value=blurry_threshold)


def task_extract_global_descriptors(**ctx) -> None:
    """
    Task 3a: Extract global descriptors (DINOv2, ISC) for sampled scenes.
    Results saved to data/processed/features/<dataset>/<scene>/global_*.npy
    """
    import mlflow
    import torch
    from scripts.data import IMC2025TrainData
    from features.engineer import FeatureEngineer, FeatureEngineeringConfig
    from features.engineer import GlobalDescriptorSpec, LocalFeatureSpec
    from pipelines.scene import Scene

    mlflow.set_tracking_uri(MLFLOW_URI)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf = FeatureEngineeringConfig(
        global_descriptors=GlobalDescriptorSpec(
            dinov2=True,
            isc=True,
            mast3r_spoc=False,  # only if weights available
            batch_size=8,
        ),
        local_features=LocalFeatureSpec(aliked=False, superpoint=False),
        output_dir=str(DATA_DIR / "processed" / "features"),
        baselines_path=str(DATA_DIR / "processed" / "eda_baselines.json"),
        log_to_mlflow=True,
        mlflow_run_name="airflow_global_descriptors",
    )

    engineer = FeatureEngineer(conf, device=device)
    data_schema = IMC2025TrainData.create(DATA_DIR)
    data_schema.preprocess()

    all_global_stats: dict = {}
    scenes_done = 0
    for (dataset, scene_name), group in data_schema.df.groupby(["dataset", "scene"]):
        if scenes_done >= 10:
            break
        image_paths = [
            data_schema.resolve_image_path(row)
            for _, row in group.iterrows()
        ]
        image_paths = [p for p in image_paths if Path(p).exists()]
        if not image_paths:
            continue

        scene = Scene(
            dataset=dataset, scene=scene_name,
            image_paths=image_paths,
            image_dir=str(Path(image_paths[0]).parent),
            data_schema=data_schema,
        )
        result = engineer.run(scene)
        all_global_stats.update(result.global_desc_stats)
        scenes_done += 1

    ctx["ti"].xcom_push(key="global_stats", value=all_global_stats)
    logger.info(f"Global descriptors extracted for {scenes_done} scenes")


def task_extract_local_features(**ctx) -> None:
    """
    Task 3b: Extract local features (ALIKED, SuperPoint) for sampled scenes.
    Results saved to data/processed/features/<dataset>/<scene>/local/<model>/
    """
    import mlflow
    import torch
    from scripts.data import IMC2025TrainData
    from features.engineer import FeatureEngineer, FeatureEngineeringConfig
    from features.engineer import GlobalDescriptorSpec, LocalFeatureSpec
    from pipelines.scene import Scene

    mlflow.set_tracking_uri(MLFLOW_URI)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf = FeatureEngineeringConfig(
        global_descriptors=GlobalDescriptorSpec(
            dinov2=False, isc=False, mast3r_spoc=False,
        ),
        local_features=LocalFeatureSpec(
            aliked=True,
            superpoint=True,
            aliked_max_keypoints=4096,
            superpoint_max_keypoints=4096,
            aliked_resize=1280,
            superpoint_resize=1600,
        ),
        output_dir=str(DATA_DIR / "processed" / "features"),
        baselines_path=str(DATA_DIR / "processed" / "eda_baselines.json"),
        log_to_mlflow=True,
        mlflow_run_name="airflow_local_features",
    )

    engineer = FeatureEngineer(conf, device=device)
    data_schema = IMC2025TrainData.create(DATA_DIR)
    data_schema.preprocess()

    scenes_done = 0
    all_local_stats: dict = {}
    for (dataset, scene_name), group in data_schema.df.groupby(["dataset", "scene"]):
        if scenes_done >= 5:
            break
        image_paths = [
            data_schema.resolve_image_path(row)
            for _, row in group.iterrows()
        ]
        image_paths = [p for p in image_paths if Path(p).exists()]
        if not image_paths:
            continue

        scene = Scene(
            dataset=dataset, scene=scene_name,
            image_paths=image_paths,
            image_dir=str(Path(image_paths[0]).parent),
            data_schema=data_schema,
        )
        result = engineer.run(scene)
        all_local_stats.update(result.local_feat_stats)
        scenes_done += 1

    ctx["ti"].xcom_push(key="local_stats", value=all_local_stats)
    logger.info(f"Local features extracted for {scenes_done} scenes")


def task_check_feature_drift(**ctx) -> None:
    """
    Task 4: Run drift detection using KS-test against EDA baselines.
    Writes drift_report.json and sets Prometheus gauges.
    Fails the task (non-zero exit) if critical drift is detected.
    """
    import mlflow
    from scripts.drift_monitor import DriftMonitor, update_prometheus_drift_metrics

    mlflow.set_tracking_uri(MLFLOW_URI)

    # Pull stats from upstream tasks via XCom
    ti = ctx["ti"]
    global_stats = ti.xcom_pull(task_ids="extract_global_descriptors",
                                 key="global_stats") or {}
    local_stats  = ti.xcom_pull(task_ids="extract_local_features",
                                 key="local_stats") or {}
    live_stats = {**global_stats, **local_stats}

    monitor = DriftMonitor(
        baselines_path=DATA_DIR / "processed" / "eda_baselines.json",
        features_dir=DATA_DIR / "processed" / "features",
    )
    report = monitor.check(
        live_stats=live_stats,
        report_path=DATA_DIR / "processed" / "drift_report.json",
    )

    # Log to MLflow
    with mlflow.start_run(run_name="drift_check", nested=True):
        mlflow.log_metrics({
            "drift_alert_count": len(report.alerts),
            "drift_status_code": {"ok": 0, "warning": 1, "critical": 2}[report.status],
        })
        mlflow.set_tag("drift_status", report.status)
        if report.alerts:
            mlflow.set_tag("drift_messages",
                           " | ".join(a.message for a in report.alerts[:3]))

    update_prometheus_drift_metrics(report)

    logger.info(
        f"Drift check complete: status={report.status}, "
        f"alerts={len(report.alerts)}"
    )

    if report.status == "critical":
        raise RuntimeError(
            "CRITICAL feature drift detected. Check drift_report.json."
        )


def task_update_dvc_metrics(**ctx) -> None:
    """
    Task 5: Write a DVC-compatible metrics JSON so `dvc metrics show`
    reflects the latest feature engineering run.
    """
    ti = ctx["ti"]
    global_stats = ti.xcom_pull(task_ids="extract_global_descriptors",
                                 key="global_stats") or {}
    local_stats  = ti.xcom_pull(task_ids="extract_local_features",
                                 key="local_stats") or {}

    drift_report_path = DATA_DIR / "processed" / "drift_report.json"
    drift_status = "ok"
    drift_alerts = 0
    if drift_report_path.exists():
        with open(drift_report_path) as f:
            dr = json.load(f)
        drift_status = dr.get("status", "ok")
        drift_alerts = len(dr.get("alerts", []))

    metrics = {
        **global_stats,
        **local_stats,
        "drift_status": drift_status,
        "drift_alert_count": drift_alerts,
    }

    out_path = DATA_DIR / "processed" / "feature_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"DVC metrics updated at {out_path}")


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="preprocess_features",
    description="Stage 2: Image preprocessing and feature engineering",
    schedule_interval="0 2 * * *",   # daily at 02:00 UTC
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    tags=["stage2", "preprocessing", "features"],
) as dag:

    t_validate = PythonOperator(
        task_id="validate_data",
        python_callable=task_validate_data,
        provide_context=True,
    )

    t_preprocess = PythonOperator(
        task_id="image_preprocess",
        python_callable=task_image_preprocess,
        provide_context=True,
    )

    t_global = PythonOperator(
        task_id="extract_global_descriptors",
        python_callable=task_extract_global_descriptors,
        provide_context=True,
    )

    t_local = PythonOperator(
        task_id="extract_local_features",
        python_callable=task_extract_local_features,
        provide_context=True,
    )

    t_drift = PythonOperator(
        task_id="check_feature_drift",
        python_callable=task_check_feature_drift,
        provide_context=True,
    )

    t_dvc = PythonOperator(
        task_id="update_dvc_metrics",
        python_callable=task_update_dvc_metrics,
        provide_context=True,
    )

    # ── Task dependencies ──────────────────────────────────────────────────
    t_validate >> t_preprocess >> [t_global, t_local] >> t_drift >> t_dvc
