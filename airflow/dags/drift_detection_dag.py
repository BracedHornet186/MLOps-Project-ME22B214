"""
drift_detection_dag.py
──────────────────────
Monitor production data drift every 30 minutes from Prometheus metrics.
If drift is detected, notify via email. NO automatic retraining.

Tasks:
  1. check_drift   – query Prometheus `feature_drift_status` (returns "drift" or "ok")
  2. branch_drift  – branch on result
  3. notify_drift  – email alert on drift
  4. end           – no-op when no drift
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/opt/airflow/project")
ALERT_EMAIL = os.environ.get("SMTP_MAIL_FROM", "mlops-team@example.com")
SMTP_USER = os.environ.get("SMTP_USER", "yashpurswani4@gmail.com")

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=10),
}


def _check_prometheus_drift(**context):
    """Query Prometheus for the feature_drift_status metric."""
    url = "http://prometheus:9090/api/v1/query?query=feature_drift_status"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10.0) as response:
            data = json.loads(response.read().decode())
            results = data.get("data", {}).get("result", [])
            if not results:
                print("No metric found in Prometheus.")
                return "ok"
            
            value_str = results[0].get("value", [0, "0"])[1]
            status_val = float(value_str)
            
            print(f"Current feature_drift_status: {status_val}")
            if status_val > 0:
                return "drift"
            return "ok"
    except Exception as e:
        print(f"Error querying Prometheus: {e}")
        return "ok"


def _decide_branch(**context):
    """Read the return code from the check_drift task."""
    ti = context["ti"]
    return_value = ti.xcom_pull(task_ids="check_drift", key="return_value")
    if return_value == "drift":
        return "notify_drift"
    return "end"


with DAG(
    dag_id="drift_detection_dag",
    description="Monitor production data drift from Prometheus and notify",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="*/30 * * * *",  # every 30 minutes
    catchup=False,
    tags=["drift", "monitoring", "prometheus"],
) as dag:

    check_drift = PythonOperator(
        task_id="check_drift",
        python_callable=_check_prometheus_drift,
    )

    branch_drift = BranchPythonOperator(
        task_id="branch_drift",
        python_callable=_decide_branch,
    )

    notify_drift = EmailOperator(
        task_id="notify_drift",
        from_email=ALERT_EMAIL,
        to=SMTP_USER,
        subject="⚠️ Data Drift Detected (Prometheus Alert)",
        html_content=(
            "<h3>⚠️ Data Drift Detected</h3>"
            "<p>The Airflow drift detection monitor has detected significant data drift "
            "based on Prometheus metrics (<code>feature_drift_status > 0</code>).</p>"
            "<p>Please review the Grafana dashboard to see the latest drift reports "
            "and trigger <code>experiment_pipeline_dag</code> manually if retraining "
            "is needed.</p>"
        ),
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success",
    )

    check_drift >> branch_drift >> [notify_drift, end]
