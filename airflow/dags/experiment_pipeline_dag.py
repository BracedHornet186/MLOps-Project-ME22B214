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
  image        : mlops-project-me22b214-ray-serve:latest   (built from Dockerfile.pipeline)
  network_mode : mlops_net                (pinned Compose default network)
  mounts       : HOST_PROJECT_ROOT → /app  (bind-mount)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.sensors.filesystem import FileSensor
from docker.types import Mount

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.environ.get("PROJECT_ROOT","/opt/airflow/project")
ALERT_EMAIL = os.environ.get("SMTP_MAIL_FROM", "mlops-team@example.com")
SMTP_USER = os.environ.get("SMTP_USER", "yashpurswani4@gmail.com")

# HOST filesystem path to the project.  Must be the real path on the Docker
# host so DockerOperator (DooD) can create a valid bind-mount.
HOST_PROJECT_ROOT = os.environ.get(
    "HOST_PROJECT_ROOT",
    "/home/abhiyaan-cu/Yash/MLOps-Project-ME22B214",
)

HOST_UID = os.environ.get('AIRFLOW_UID', '1000')
HOST_GID = os.environ.get('DOCKER_GID', '984')

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# The Compose default network is pinned to this name in docker-compose.yaml.
# Ephemeral pipeline containers join it so they can reach `mlflow:5000`.
COMPOSE_NETWORK = "mlops_net"

PIPELINE_IMAGE = "mlops-project-me22b214-ray-serve:latest"

# Bind-mount: host project dir → /app inside the pipeline container.
# .dvc/, dvc.yaml, scripts/, conf/, data/ are all present under this path.
_PROJECT_MOUNT = Mount(
    source=HOST_PROJECT_ROOT,
    target="/app",
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
        from_email=ALERT_EMAIL,
        to=SMTP_USER,
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
                f"trap 'chown -R {HOST_UID}:{HOST_GID} /app' EXIT; "
                "set -e && cd /app && "
                "export PYTHONPATH=/app && "
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

    # ── Best-run promotion ─────────────────────────────────────────────────────

    select_best_run = BashOperator(
        task_id="select_best_run",
        bash_command=f"cd {PROJECT_ROOT} && python scripts/select_best_run.py",
    )

    notify_user = EmailOperator(
        task_id="notify_user",
        to=SMTP_USER,
        from_email=ALERT_EMAIL,
        subject="Production Config Updated",
        html_content=(
            "<h3>Production Config Updated</h3>"
            "<p>A new best run has been promoted to production.</p>"
            f"<p>Config written to: <code>{PROJECT_ROOT}/conf/best_config.yaml</code></p>"
        ),
    )

    # ── Dependencies ──────────────────────────────────────────────────────────

    sensors = [wait_for_train_dir, wait_for_train_labels, wait_for_train_thresholds]

    sensors >> notify_missing_data >> end_no_data
    sensors >> run_dvc_pipeline >> select_best_run >> notify_user
