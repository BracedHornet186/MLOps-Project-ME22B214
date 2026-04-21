"""
experiment_pipeline_dag.py
──────────────────────────
Run the full DVC experiment pipeline with MLflow parent-run tracking.

Tasks:
  1. wait_for_* (FileSensors)   – guard: required data files must be present
  2. notify_missing_data        – email if any sensor times out (soft_fail)
  3. run_dvc_pipeline           – create MLflow parent run, execute dvc repro,
                                  then close the parent run
  4. select_best_run            – promote the best mAA run to production
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.sensors.filesystem import FileSensor
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.providers.standard.operators.empty import EmptyOperator

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/opt/airflow/project")
ALERT_EMAIL = os.environ.get("ALERT_EMAIL", "mlops-team@example.com")

PYTHON = f"{PROJECT_ROOT}/.venv/bin/python3"
PYTHON_CMD = f'if [ -x "{PYTHON}" ]; then PY="{PYTHON}"; else PY="python3"; fi'

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=6),
}

with DAG(
    dag_id="experiment_pipeline_dag",
    description="Run DVC experiments with MLflow tracking",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,  # Manual trigger only
    catchup=False,
    tags=["experiment", "dvc", "mlflow"],
) as dag:

    # ── Data availability guards ─────────────────────────────────────────────

    wait_for_train_dir = FileSensor(
        task_id="wait_for_train_dir",
        filepath=f"{PROJECT_ROOT}/data/train",
        poke_interval=60,
        timeout=300,
        mode="reschedule",
        soft_fail=True,
    )

    wait_for_train_labels = FileSensor(
        task_id="wait_for_train_labels",
        filepath=f"{PROJECT_ROOT}/data/train_labels.csv",
        poke_interval=60,
        timeout=300,
        mode="reschedule",
        soft_fail=True,
    )

    wait_for_train_thresholds = FileSensor(
        task_id="wait_for_train_thresholds",
        filepath=f"{PROJECT_ROOT}/data/train_thresholds.csv",
        poke_interval=60,
        timeout=300,
        mode="reschedule",
        soft_fail=True,
    )

    notify_missing_data = EmailOperator(
        task_id="notify_missing_data",
        to=ALERT_EMAIL,
        subject="⚠️ Experiment pipeline: required data files missing",
        html_content=(
            "<h3>⚠️ Pipeline cannot start</h3>"
            "<p>One or more required data files are missing:</p>"
            "<ul>"
            f"<li><code>{PROJECT_ROOT}/data/train/</code></li>"
            f"<li><code>{PROJECT_ROOT}/data/train_labels.csv</code></li>"
            f"<li><code>{PROJECT_ROOT}/data/train_thresholds.csv</code></li>"
            "</ul>"
            "<p>Please upload the data and re-trigger "
            "<code>experiment_pipeline_dag</code>.</p>"
        ),
        trigger_rule="one_failed",
    )

    # ── Skip gate: do nothing if sensors succeeded but we still reach end ───

    end_no_data = EmptyOperator(
        task_id="end_no_data",
        trigger_rule="none_failed_min_one_success",
    )

    # ── DVC pipeline ─────────────────────────────────────────────────────────

    run_dvc_pipeline = BashOperator(
        task_id="run_dvc_pipeline",
        bash_command=(
            f"set -e; cd {PROJECT_ROOT}; "
            f"{PYTHON_CMD}; "
            # Create ONE parent run (script no longer uses `with` block so run stays ACTIVE)
            "PARENT_ID=$($PY scripts/start_parent_dvc_run.py | tail -n 1); "
            'echo "MLflow parent run: $PARENT_ID"; '
            # Run all DVC stages nested under the parent run
            'MLFLOW_PARENT_RUN_ID="$PARENT_ID" dvc repro; '
            # Close the parent run cleanly after dvc repro finishes
            "$PY scripts/end_parent_dvc_run.py \"$PARENT_ID\""
        ),
        trigger_rule="all_success",
    )

    select_best_run = BashOperator(
        task_id="select_best_run",
        bash_command=(
            f"set -e; cd {PROJECT_ROOT}; "
            f"{PYTHON_CMD}; "
            "$PY scripts/select_best_run.py"
        ),
        do_xcom_push=True,
    )

    # ── Dependencies ─────────────────────────────────────────────────────────

    sensors = [wait_for_train_dir, wait_for_train_labels, wait_for_train_thresholds]

    sensors >> notify_missing_data >> end_no_data
    sensors >> run_dvc_pipeline >> select_best_run
