from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator

PROJECT_ROOT = Path("/opt/airflow/project")


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
    schedule_interval=None,
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
        to="mlops-team@example.com",
        subject="Production Config Updated",
        html_content="New production config deployed",
    )

    select_best_run >> notify_user
