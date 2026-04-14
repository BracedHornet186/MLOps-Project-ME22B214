"""
dags/scene_reconstruction_dag.py
────────────────────────────────────────────────────────────────────────────
Airflow DAG covering Stage 3 (experiment tracking) and Stage 4 (inference).

DAG graph
─────────
validate_data → eda_baselines → run_experiment → register_model
                                               ↘ notify_if_degraded

Schedule: @weekly (Sunday 02:00 UTC) for automated re-evaluation.
Can also be triggered manually from the Airflow UI.

Tasks
─────
1. validate_data      Run scripts/validate_data.py, assert no schema errors
2. eda_baselines      Recompute EDA baselines, log to MLflow eda_baselines exp
3. run_experiment     Run train_experiment.py with RTX 3060 config
4. register_model     Promote to Production if mAA beats current champion
5. notify_if_degraded Alert (log warning) if mAA < 0.45 threshold
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

log = logging.getLogger(__name__)

# ── Paths inside the Airflow container ───────────────────────────────────────
PROJECT_ROOT = Path("/opt/airflow/project")
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
CONF_DIR     = PROJECT_ROOT.parent / "conf"  # mounted at repo root

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MAA_THRESHOLD = 0.45   # alert if overall mAA drops below this

# ─────────────────────────────────────────────────────────────────────────────
# Default args
# ─────────────────────────────────────────────────────────────────────────────

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# Task functions
# ─────────────────────────────────────────────────────────────────────────────

def task_validate_data(**context) -> None:
    """Run validate_data.py and assert no schema errors."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "validate_data.py")],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )
    log.info("validate_data stdout:\n%s", result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Data validation failed:\n{result.stderr}")

    report_path = PROJECT_ROOT / "data/processed/validation_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        log.info("Validation report: %s", json.dumps(report, indent=2))
        # Push report to XCom so downstream tasks can read it
        context["ti"].xcom_push(key="validation_report", value=report)


def task_eda_baselines(**context) -> None:
    """Recompute EDA baselines and log to MLflow."""
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("eda_baselines")

    baselines_path = PROJECT_ROOT / "data/processed/eda_baselines.json"
    if not baselines_path.exists():
        log.warning("eda_baselines.json not found — skipping EDA log")
        return

    baselines = json.loads(baselines_path.read_text())

    with mlflow.start_run(run_name=f"eda_{datetime.utcnow().strftime('%Y%m%d')}"):
        flat = {
            "total_images":       baselines.get("dataset", {}).get("total_images", 0),
            "total_scenes":       baselines.get("dataset", {}).get("total_scenes", 0),
            "sharpness_p10":      baselines.get("sharpness", {}).get("p10", 0),
            "desc_norm_mean":     baselines.get("descriptor", {}).get("norm_mean", 0),
            "desc_norm_std":      baselines.get("descriptor", {}).get("norm_std", 0),
            "non_upright_pct":    baselines.get("orientation", {}).get("non_upright_pct", 0),
        }
        mlflow.log_metrics(flat)
        mlflow.log_artifact(str(baselines_path), artifact_path="baselines")
        log.info("EDA baselines logged: %s", flat)


def task_run_experiment(**context) -> None:
    """
    Run the full inference pipeline on the training split and log to MLflow.
    Pushes the resulting mAA to XCom for downstream tasks.
    """
    import mlflow
    from scripts.config import PipelineConfig, SubmissionConfig
    from scripts.data import DEFAULT_DATASET_DIR, IMC2025TrainData
    from scripts.kernel import run_and_save_submission
    from scripts.distributed import DistConfig
    import utils.imc25.metric as imc25_metric
    import pandas as pd
    import time

    config_path = CONF_DIR / "pipeline/imc2025/mast3r_rtx3060.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("scene_reconstruction")

    pipeline_conf = PipelineConfig.load_config(config_path)
    submission_conf = SubmissionConfig(
        pipeline=pipeline_conf,
        target_data_type="imc2025train",
    )

    run_name = f"weekly_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_params({
            "config_file": config_path.name,
            "pipeline_type": "imc2025_mast3r",
            "schedule": "weekly",
        })

        t_start = time.perf_counter()
        submission_csv = PROJECT_ROOT.parent / "submission.csv"
        run_and_save_submission(
            submission_conf,
            env_name="local",
            data_root_dir=DEFAULT_DATASET_DIR,
            dist_conf=DistConfig.single(),
        )
        elapsed = time.perf_counter() - t_start

        # Score against ground truth
        final_score, dataset_scores = imc25_metric.score(
            gt_csv=DEFAULT_DATASET_DIR / "train_labels.csv",
            user_csv=submission_csv,
            thresholds_csv=DEFAULT_DATASET_DIR / "train_thresholds.csv",
            mask_csv=None,
            inl_cf=0, strict_cf=-1, verbose=True,
        )

        mlflow.log_metric("mAA_overall", final_score)
        for ds, sc in dataset_scores.items():
            mlflow.log_metric(f"mAA_{ds}", float(sc))
        mlflow.log_metrics({
            "inference_latency_seconds": round(elapsed, 2),
        })
        mlflow.log_artifact(str(config_path), artifact_path="config")
        if submission_csv.exists():
            mlflow.log_artifact(str(submission_csv), artifact_path="predictions")

        log.info(
            "Experiment done — mAA=%.4f  elapsed=%.1fs  run_id=%s",
            final_score, elapsed, mlflow_run_id,
        )

    context["ti"].xcom_push(key="maa_overall", value=float(final_score))
    context["ti"].xcom_push(key="mlflow_run_id", value=mlflow_run_id)


def task_register_model(**context) -> None:
    """Promote the current run to Production in MLflow Model Registry if best."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    run_id  = context["ti"].xcom_pull(task_ids="run_experiment", key="mlflow_run_id")
    new_maa = context["ti"].xcom_pull(task_ids="run_experiment", key="maa_overall")

    if not run_id:
        log.warning("No run_id found — skipping model registration")
        return

    model_name = "scene_reconstruction_mast3r_rtx3060"
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URI)

    # Check Production champion
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            prod_maa = client.get_run(prod_versions[0].run_id).data.metrics.get("mAA_overall", 0.0)
            stage = "Production" if new_maa > prod_maa else "Staging"
            log.info("New mAA=%.4f vs Production=%.4f → stage=%s", new_maa, prod_maa, stage)
        else:
            stage = "Production"
    except Exception:
        stage = "Production"

    model_uri = f"runs:/{run_id}/config"
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=registered.version,
        stage=stage,
    )
    log.info("Registered %s v%s → %s", model_name, registered.version, stage)


def task_branch_on_maa(**context) -> str:
    """Route to notify_degraded if mAA < threshold, else to end."""
    maa = context["ti"].xcom_pull(task_ids="run_experiment", key="maa_overall") or 0.0
    log.info("Branching on mAA=%.4f (threshold=%.2f)", maa, MAA_THRESHOLD)
    return "notify_degraded" if maa < MAA_THRESHOLD else "end"


def task_notify_degraded(**context) -> None:
    """Log a critical warning when mAA drops below threshold."""
    maa = context["ti"].xcom_pull(task_ids="run_experiment", key="maa_overall") or 0.0
    run_id = context["ti"].xcom_pull(task_ids="run_experiment", key="mlflow_run_id")
    msg = (
        f"ALERT: mAA={maa:.4f} is below threshold {MAA_THRESHOLD}. "
        f"MLflow run: {MLFLOW_URI}/#/experiments (run_id={run_id}). "
        "Review config or promote a better Staging model to Production."
    )
    log.critical(msg)
    # In production: send to Slack/PagerDuty here. e.g.:
    # requests.post(SLACK_WEBHOOK_URL, json={"text": msg})


# ─────────────────────────────────────────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="scene_reconstruction_pipeline",
    description="Stage 3+4: Experiment tracking + model registration (weekly)",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 2 * * 0",    # Sunday 02:00 UTC
    catchup=False,
    tags=["scene_reconstruction", "mlops", "stage3", "stage4"],
    max_active_runs=1,                 # one run at a time — single GPU
) as dag:

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=task_validate_data,
    )

    eda = PythonOperator(
        task_id="eda_baselines",
        python_callable=task_eda_baselines,
    )

    experiment = PythonOperator(
        task_id="run_experiment",
        python_callable=task_run_experiment,
        execution_timeout=timedelta(hours=3),   # max 3h for large datasets
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=task_register_model,
    )

    branch = BranchPythonOperator(
        task_id="branch_on_maa",
        python_callable=task_branch_on_maa,
    )

    notify = PythonOperator(
        task_id="notify_degraded",
        python_callable=task_notify_degraded,
    )

    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    # DAG edges
    validate >> eda >> experiment >> register >> branch >> [notify, end]
    notify >> end
