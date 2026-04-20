"""
model_selection_dag.py
──────────────────────
Periodic model selection: pick the best run by mAA and promote it.

Tasks:
  1. select_best_run – run scripts/select_best_run.py
  2. notify_user     – email notification with best mAA and run_id
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/home/abhiyaan-cu/Yash/MLOps-Project-ME22B214")
ALERT_EMAIL = os.environ.get("ALERT_EMAIL", "mlops-team@example.com")

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(minutes=15),
}

with DAG(
    dag_id="model_selection_dag",
    description="Select best run based on mAA and promote best config",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 */6 * * *",  # every 6 hours
    catchup=False,
    tags=["model-selection", "mlflow"],
) as dag:

    select_best_run = BashOperator(
        task_id="select_best_run",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            ".venv/bin/python3 scripts/select_best_run.py"
        ),
    )

    notify_user = EmailOperator(
        task_id="notify_user",
        to=ALERT_EMAIL,
        subject="New Best Model Selected",
        html_content=(
            "<h3>Model Selection Complete</h3>"
            "<p>The best run has been selected based on mAA score "
            "and its config has been promoted to <code>conf/best_config.yaml</code>.</p>"
            "<p>Check MLflow at <a href='http://localhost:5000'>localhost:5000</a> for details.</p>"
        ),
    )

    select_best_run >> notify_user
