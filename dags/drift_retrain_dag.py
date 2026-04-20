"""
dags/drift_retrain_dag.py
────────────────────────────────────────────────────────────────────────────
Automated Retraining DAG — triggered by data drift or performance decay.

DAG graph
─────────
  run_drift_check → branch_on_drift → [skip_retrain, run_retrain]
                                                │
                                          register_model → send_notification

Schedule: @weekly (Wednesday 03:00 UTC) as safety net.
Can also be triggered manually from the Airflow UI,
or via Alertmanager webhook (POST /api/v1/dags/drift_retrain_pipeline/dagRuns).

Tasks
─────
1. run_drift_check      Run full drift detection (input quality + performance)
2. branch_on_drift      If drift detected → retrain, else → skip
3. run_retrain          Run train_experiment.py with mast3r.yaml config
4. register_model       Promote to Production if mAA beats current champion
5. skip_retrain         No-op when no drift detected
6. send_notification    Email notification with results via EmailOperator
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.email import EmailOperator

log = logging.getLogger(__name__)

# ── Paths inside the Airflow container ───────────────────────────────────────
PROJECT_ROOT = Path("/opt/airflow/project")
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
CONF_DIR     = PROJECT_ROOT.parent / "conf"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DATA_DIR     = Path(os.environ.get("DEFAULT_DATASET_DIR", f"{PROJECT_ROOT}/data"))
ALERT_EMAIL  = os.environ.get("ALERT_EMAIL", "mlops-team@example.com")

# ─────────────────────────────────────────────────────────────────────────────
# Default args
# ─────────────────────────────────────────────────────────────────────────────

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
    "email_on_retry": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# Task functions
# ─────────────────────────────────────────────────────────────────────────────

def task_run_drift_check(**context) -> None:
    """
    Run full drift detection: input quality (descriptor, sharpness,
    brightness, contrast) + performance proxy (mAA / registration_rate).
    """
    from scripts.drift_monitor import DriftMonitor, update_prometheus_drift_metrics

    monitor = DriftMonitor(
        baselines_path=DATA_DIR / "processed" / "eda_baselines.json",
        features_dir=DATA_DIR / "processed" / "features",
        mlflow_uri=MLFLOW_URI,
    )
    report = monitor.check(
        report_path=DATA_DIR / "processed" / "drift_report.json",
        check_performance=True,
    )

    update_prometheus_drift_metrics(report)

    # Push drift report to XCom
    context["ti"].xcom_push(key="drift_status", value=report.status)
    context["ti"].xcom_push(key="drift_alert_count", value=len(report.alerts))
    context["ti"].xcom_push(
        key="drift_messages",
        value=[a.message for a in report.alerts[:5]],
    )

    log.info(
        "Drift check complete: status=%s, alerts=%d",
        report.status, len(report.alerts),
    )


def task_branch_on_drift(**context) -> str:
    """Route to retrain if drift detected, else skip."""
    drift_status = context["ti"].xcom_pull(
        task_ids="run_drift_check", key="drift_status"
    ) or "ok"
    log.info("Branching on drift_status=%s", drift_status)
    if drift_status in ("warning", "critical"):
        return "run_retrain"
    return "skip_retrain"


def task_run_retrain(**context) -> None:
    """
    Re-run the inference pipeline with current config and log to MLflow.
    This acts as a hyperparameter re-evaluation on the existing data.
    """
    from scripts.train_experiment import run_experiment

    config_path = CONF_DIR / "mast3r.yaml"
    if not config_path.exists():
        # Fallback to the standard mast3r.yaml
        config_path = CONF_DIR / "mast3r.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    t_start = time.perf_counter()
    run_id = run_experiment(
        config_path=str(config_path),
        datasets=None,  # evaluate all training datasets
        experiment_name="scene_reconstruction",
        run_name=f"drift_retrain_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
        mlflow_uri=MLFLOW_URI,
        register_if_best=True,
    )
    elapsed = time.perf_counter() - t_start

    # Pull metrics from the run for downstream notification
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URI)
    run_data = client.get_run(run_id)
    new_maa = run_data.data.metrics.get("mAA_overall", 0.0)
    new_reg_rate = run_data.data.metrics.get("registration_rate", 0.0)

    context["ti"].xcom_push(key="retrain_run_id", value=run_id)
    context["ti"].xcom_push(key="retrain_maa", value=new_maa)
    context["ti"].xcom_push(key="retrain_reg_rate", value=new_reg_rate)
    context["ti"].xcom_push(key="retrain_elapsed", value=round(elapsed, 1))
    context["ti"].xcom_push(key="retrain_config", value=str(config_path.name))

    log.info(
        "Retrain complete — mAA=%.4f  reg_rate=%.2f%%  elapsed=%.1fs  run_id=%s",
        new_maa, 100 * new_reg_rate, elapsed, run_id,
    )


def task_register_model(**context) -> None:
    """Promote the retrained model to Production if it beats the champion."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    run_id = context["ti"].xcom_pull(task_ids="run_retrain", key="retrain_run_id")
    new_maa = context["ti"].xcom_pull(task_ids="run_retrain", key="retrain_maa") or 0.0

    if not run_id:
        log.warning("No run_id from retrain — skipping registration")
        return

    model_name = "scene_reconstruction_mast3r"
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URI)

    # Check Production champion
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            prod_maa = client.get_run(
                prod_versions[0].run_id
            ).data.metrics.get("mAA_overall", 0.0)
            if new_maa > prod_maa:
                stage = "Production"
                log.info(
                    "New mAA=%.4f beats Production=%.4f → promoting",
                    new_maa, prod_maa,
                )
            else:
                stage = "Staging"
                log.info(
                    "New mAA=%.4f does NOT beat Production=%.4f → Staging only",
                    new_maa, prod_maa,
                )
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

    context["ti"].xcom_push(key="promoted_stage", value=stage)
    log.info("Registered %s v%s → %s", model_name, registered.version, stage)


def _build_email_body(**context) -> str:
    """Build the email body from XCom values."""
    ti = context["ti"]
    drift_status = ti.xcom_pull(task_ids="run_drift_check", key="drift_status") or "ok"
    drift_alerts = ti.xcom_pull(task_ids="run_drift_check", key="drift_alert_count") or 0
    drift_msgs = ti.xcom_pull(task_ids="run_drift_check", key="drift_messages") or []

    run_id = ti.xcom_pull(task_ids="run_retrain", key="retrain_run_id")
    maa = ti.xcom_pull(task_ids="run_retrain", key="retrain_maa")
    reg_rate = ti.xcom_pull(task_ids="run_retrain", key="retrain_reg_rate")
    elapsed = ti.xcom_pull(task_ids="run_retrain", key="retrain_elapsed")
    config = ti.xcom_pull(task_ids="run_retrain", key="retrain_config")
    stage = ti.xcom_pull(task_ids="register_model", key="promoted_stage") or "N/A"

    lines = [
        "=" * 60,
        "  DRIFT-TRIGGERED RETRAINING REPORT",
        "=" * 60,
        "",
        f"Drift Status     : {drift_status.upper()}",
        f"Drift Alerts     : {drift_alerts}",
    ]
    if drift_msgs:
        lines.append("Drift Messages   :")
        for msg in drift_msgs:
            lines.append(f"  - {msg}")
    lines += [
        "",
        "--- Retraining Results ---",
        f"Config           : {config or 'N/A'}",
        f"MLflow Run ID    : {run_id or 'N/A'}",
        f"mAA Overall      : {maa:.4f}" if maa is not None else "mAA Overall      : N/A",
        f"Registration Rate: {100*reg_rate:.1f}%" if reg_rate is not None else "Registration Rate: N/A",
        f"Elapsed Time     : {elapsed}s" if elapsed is not None else "Elapsed Time     : N/A",
        f"Promoted to      : {stage}",
        "",
        f"MLflow UI        : {MLFLOW_URI}",
        "=" * 60,
    ]
    return "\n".join(lines)


def task_prepare_email(**context) -> None:
    """Prepare email content and push to XCom for EmailOperator."""
    body = _build_email_body(**context)
    context["ti"].xcom_push(key="email_body", value=body)

    drift_status = context["ti"].xcom_pull(
        task_ids="run_drift_check", key="drift_status"
    ) or "ok"
    maa = context["ti"].xcom_pull(task_ids="run_retrain", key="retrain_maa")
    subject = (
        f"[MLOps] Drift Retrain Complete — "
        f"status={drift_status.upper()}, mAA={maa:.4f}" if maa is not None
        else f"[MLOps] Drift Retrain Complete — status={drift_status.upper()}"
    )
    context["ti"].xcom_push(key="email_subject", value=subject)


# ─────────────────────────────────────────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="drift_retrain_pipeline",
    description="Automated retraining triggered by data drift or performance decay",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 3 * * 3",    # Wednesday 03:00 UTC
    catchup=False,
    tags=["drift", "retrain", "mlops", "stage5"],
    max_active_runs=1,                 # single GPU — one run at a time
) as dag:

    drift_check = PythonOperator(
        task_id="run_drift_check",
        python_callable=task_run_drift_check,
    )

    branch = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=task_branch_on_drift,
    )

    skip = EmptyOperator(
        task_id="skip_retrain",
    )

    retrain = PythonOperator(
        task_id="run_retrain",
        python_callable=task_run_retrain,
        execution_timeout=timedelta(hours=4),
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=task_register_model,
    )

    prepare_email = PythonOperator(
        task_id="prepare_email",
        python_callable=task_prepare_email,
    )

    send_email = EmailOperator(
        task_id="send_notification",
        to=ALERT_EMAIL,
        subject="{{ ti.xcom_pull(task_ids='prepare_email', key='email_subject') }}",
        html_content="<pre>{{ ti.xcom_pull(task_ids='prepare_email', key='email_body') }}</pre>",
        trigger_rule="none_failed_min_one_success",
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success",
    )

    # DAG edges
    drift_check >> branch >> [skip, retrain]
    retrain >> register >> prepare_email >> send_email >> end
    skip >> end
