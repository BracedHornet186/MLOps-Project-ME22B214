"""
drift_detection_dag.py
──────────────────────
Monitor production data drift every 30 minutes.
If drift is detected, notify via email. NO automatic retraining.

Tasks:
  1. check_drift   – run detect_drift.py (exit 0=ok, 1=drift)
  2. branch_drift  – branch on result
  3. notify_drift  – email alert on drift
  4. end           – no-op when no drift
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator

HOST_PROJECT_ROOT = os.environ.get("HOST_PROJECT_ROOT", "/home/abhiyaan-cu/Yash/MLOps-Project-ME22B214")
ALERT_EMAIL = os.environ.get("SMTP_MAIL_FROM", "mlops-team@example.com")
SMTP_USER = os.environ.get("SMTP_USER", "yashpurswani4@gmail.com")

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=10),
}


def _decide_branch(**context):
    """Read the return code from the check_drift task."""
    ti = context["ti"]
    return_value = ti.xcom_pull(task_ids="check_drift", key="return_value")
    # detect_drift.py pushes "drift" or "ok" via xcom
    if return_value == "drift":
        return "notify_drift"
    return "end"


with DAG(
    dag_id="drift_detection_dag",
    description="Monitor production data drift and notify on detection",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="*/30 * * * *",  # every 30 minutes
    catchup=False,
    tags=["drift", "monitoring"],
) as dag:

    check_drift = BashOperator(
        task_id="check_drift",
        bash_command=(
            f"cd {HOST_PROJECT_ROOT} && "
            "python scripts/detect_drift.py"
        ),
        do_xcom_push=True,
    )

    branch_drift = BranchPythonOperator(
        task_id="branch_drift",
        python_callable=_decide_branch,
    )

    notify_drift = EmailOperator(
        task_id="notify_drift",
        from_email=ALERT_EMAIL,
        to=SMTP_USER,
        subject="⚠️ Data Drift Detected",
        html_content=(
            "<h3>⚠️ Data Drift Detected</h3>"
            "<p>The drift detection monitor has detected significant data drift.</p>"
            "<p>Please review the drift report at "
            f"<code>{HOST_PROJECT_ROOT}/data/processed/drift_report.json</code> "
            "and trigger <code>experiment_pipeline_dag</code> manually if retraining "
            "is needed.</p>"
        ),
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success",
    )

    check_drift >> branch_drift >> [notify_drift, end]
