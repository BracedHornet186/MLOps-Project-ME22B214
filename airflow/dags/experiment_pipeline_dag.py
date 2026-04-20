"""
experiment_pipeline_dag.py
──────────────────────────
Run the full DVC experiment pipeline with MLflow parent-run tracking.

Tasks:
  1. start_parent_run  – create an MLflow parent run, capture run_id
  2. run_dvc_pipeline  – execute `dvc repro` under that parent run
  3. select_best_run   – promote the best mAA run to production
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/home/abhiyaan-cu/Yash/MLOps-Project-ME22B214")

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=3),
}

with DAG(
    dag_id="experiment_pipeline_dag",
    description="Run DVC experiments with MLflow tracking",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["experiment", "dvc", "mlflow"],
) as dag:

    start_parent_run = BashOperator(
        task_id="start_parent_run",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "PARENT_ID=$(.venv/bin/python3 scripts/start_parent_dvc_run.py | head -n 1) && "
            'echo "PARENT_ID=$PARENT_ID"'
        ),
    )

    run_dvc_pipeline = BashOperator(
        task_id="run_dvc_pipeline",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "PARENT_ID=$(.venv/bin/python3 scripts/start_parent_dvc_run.py | head -n 1) && "
            'MLFLOW_PARENT_RUN_ID="$PARENT_ID" dvc repro'
        ),
    )

    select_best_run = BashOperator(
        task_id="select_best_run",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            ".venv/bin/python3 scripts/select_best_run.py"
        ),
    )

    start_parent_run >> run_dvc_pipeline >> select_best_run
