"""
deploy_production_dag.py
──────────────────────
Periodic model selection: pick the best run by mAA and promote it.

Tasks:
  1. select_best_run – run scripts/select_best_run.py
  2. notify_user     – email notification with best mAA and run_id
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import os
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.smtp.operators.smtp import EmailOperator

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/opt/airflow/project")
ALERT_EMAIL = os.environ.get("ALERT_EMAIL", "mlops-team@example.com")

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}


with DAG(
    dag_id="deploy_production_dag",
    description="Deploy latest best production config for FastAPI serving",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="@hourly",
    catchup=False,
    tags=["deployment", "production", "mlflow"],
) as dag:
    select_best_run = BashOperator(
        task_id="select_best_run",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "if [ -x .venv/bin/python3 ]; then "
            ".venv/bin/python3 scripts/select_best_run.py; "
            "else "
            "python3 scripts/select_best_run.py; "
            "fi"
        ),
    )

    notify_user = EmailOperator(
        task_id="notify_user",
        to=ALERT_EMAIL,
        subject="Production Config Updated",
        html_content=(
            "<h3>Production Config Updated</h3>"
            "<p>A new best run has been promoted to production.</p>"
            "<pre>{{ ti.xcom_pull(task_ids='select_best_run') }}</pre>"
            f"<p>Config written to: <code>{PROJECT_ROOT}/conf/best_config.yaml</code></p>"
        ),
    )

    select_best_run >> notify_user
