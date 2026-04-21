"""
experiment_pipeline_dag.py
──────────────────────────
Run the full DVC experiment pipeline inside an ephemeral Docker container.

Tasks:
  1. wait_for_*          – FileSensors guard required data files
  2. notify_missing_data – email alert when sensors time out
  3. run_dvc_pipeline    – DockerOperator: starts MLflow parent run,
                           executes `dvc repro`, closes parent run
  4. select_best_run     – DockerOperator: promotes best mAA run to production

DockerOperator details:
  image        : my-dvc-pipeline:latest   (built from Dockerfile.pipeline)
  network_mode : mlops_net                (pinned Compose default network)
  mounts       : HOST_PROJECT_ROOT → /project  (bind-mount)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.sensors.filesystem import FileSensor
from docker.types import Mount

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/opt/airflow/project")
ALERT_EMAIL  = os.environ.get("ALERT_EMAIL",  "mlops-team@example.com")

# HOST filesystem path to the project.  Must be the real path on the Docker
# host so DockerOperator (DooD) can create a valid bind-mount.
HOST_PROJECT_ROOT = os.environ.get(
    "HOST_PROJECT_ROOT",
    "/home/abhiyaan-cu/Yash/MLOps-Project-ME22B214",
)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# The Compose default network is pinned to this name in docker-compose.yaml.
# Ephemeral pipeline containers join it so they can reach `mlflow:5000`.
COMPOSE_NETWORK = "mlops_net"

PIPELINE_IMAGE = "my-dvc-pipeline:latest"

# Bind-mount: host project dir → /project inside the pipeline container.
# .dvc/, dvc.yaml, scripts/, conf/, data/ are all present under this path.
_PROJECT_MOUNT = Mount(
    source=HOST_PROJECT_ROOT,
    target="/project",
    type="bind",
)

# ── DAG defaults ──────────────────────────────────────────────────────────────

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=6),
}

# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="experiment_pipeline_dag",
    description="Run DVC experiments with MLflow tracking (DockerOperator)",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,   # manual trigger only
    catchup=False,
    tags=["experiment", "dvc", "mlflow", "docker"],
) as dag:

    # ── Data availability guards ──────────────────────────────────────────────

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
            "<p>One or more required data files are missing under "
            f"<code>{PROJECT_ROOT}/data/</code>:</p>"
            "<ul>"
            "<li><code>train/</code></li>"
            "<li><code>train_labels.csv</code></li>"
            "<li><code>train_thresholds.csv</code></li>"
            "</ul>"
            "<p>Upload the data then re-trigger "
            "<code>experiment_pipeline_dag</code>.</p>"
        ),
        trigger_rule="one_failed",
    )

    end_no_data = EmptyOperator(
        task_id="end_no_data",
        trigger_rule="none_failed_min_one_success",
    )

    # ── DVC pipeline (Docker) ─────────────────────────────────────────────────

    run_dvc_pipeline = DockerOperator(
        task_id="run_dvc_pipeline",
        image=PIPELINE_IMAGE,
        command=[
            "bash", "-c",
            (
                "set -e && cd /project && "
                "export PYTHONPATH=/project && "
                "python3 scripts/run_pipeline_with_parent.py"   
            ),
        ],
        environment={
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "MLFLOW_EXPERIMENT_NAME": "scene_reconstruction_dvc",
        },
        device_requests=[{"count": -1, "capabilities": [["gpu"]]}],
        mounts=[_PROJECT_MOUNT],
        network_mode=COMPOSE_NETWORK,
        auto_remove="force",
        mount_tmp_dir=False,
        docker_url="unix:///var/run/docker.sock",
        trigger_rule="all_success",
    )

    # ── Best-run promotion (Docker) ───────────────────────────────────────────

    select_best_run = DockerOperator(
        task_id="select_best_run",
        image=PIPELINE_IMAGE,
        command=[
            "bash", "-c",
            "set -e && cd /project && python3 scripts/select_best_run.py",
        ],
        environment={"MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI},
        mounts=[_PROJECT_MOUNT],
        network_mode=COMPOSE_NETWORK,
        auto_remove="force",
        mount_tmp_dir=False,
        docker_url="unix:///var/run/docker.sock",
    )

    # ── Dependencies ──────────────────────────────────────────────────────────

    sensors = [wait_for_train_dir, wait_for_train_labels, wait_for_train_thresholds]

    sensors >> notify_missing_data >> end_no_data
    sensors >> run_dvc_pipeline >> select_best_run
