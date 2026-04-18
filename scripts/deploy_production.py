#!/usr/bin/env python3
"""
scripts/deploy_production.py
────────────────────────────────────────────────────────────────────────────
Deployment Script

Fetches the 'Production' run configuration from MLflow Registry and applies
it to the current host deployment (by overwriting conf/mast3r.yaml), then
restarts the model-server and api Docker containers to load the new config.

Usage:
    python3 scripts/deploy_production.py
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking.client import MlflowClient

# ── Setup ───────────────────────────────────────────────────────────────────

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Constants
ROOT = Path(__file__).resolve().parent.parent
CONF_DIR = ROOT / "conf"
ACTIVE_CONFIG = CONF_DIR / "mast3r.yaml"

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "scene_reconstruction_mast3r"
TARGET_STAGE = "Production"

def main():
    log.info("Starting deployment process...")
    log.info(f"Targeting MLflow server at: {MLFLOW_URI}")

    # 1. Connect to MLFlow and find Production model
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient(tracking_uri=MLFLOW_URI)

    try:
        log.info(f"Searching for model '{MODEL_NAME}' in stage '{TARGET_STAGE}'...")
        versions = client.get_latest_versions(MODEL_NAME, stages=[TARGET_STAGE])
        if not versions:
            log.error(f"No model found in '{TARGET_STAGE}' stage for '{MODEL_NAME}'. Aborting deploy.")
            return

        prod_version = versions[0]
        run_id = prod_version.run_id
        version_num = prod_version.version
        log.info(f"Found Production Model Version: {version_num} (Run ID: {run_id})")

        # 2. Download 'config' artifact
        # Notice: the train_experiment.py logs the artifact as 'config' or 'config/mast3r.yaml' etc.
        # We need to list artifacts to find the exact filename logged, usually just the file inside 'config' directory.
        artifacts = client.list_artifacts(run_id, path="config")
        yaml_artifacts = [a for a in artifacts if a.path.endswith(".yaml")]
        if not yaml_artifacts:
            log.error("Could not find a .yaml config artifact in the Production run. Aborting.")
            return
            
        config_artifact = yaml_artifacts[0]
        log.info(f"Downloading configuration artifact: {config_artifact.path}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = client.download_artifacts(run_id, config_artifact.path, dst_path=temp_dir)
            
            # 3. Overwrite host conf/mast3r.yaml
            log.info(f"Updating local configuration: {ACTIVE_CONFIG}")
            CONF_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(temp_path, ACTIVE_CONFIG)

    except Exception as e:
        log.error(f"Failed to fetch model from MLflow: {e}")
        return

    # 4. Restart Docker containers
    log.info("Restarting docker containers: model-server and api...")
    try:
        # Determine whether to use 'docker compose' or 'docker-compose'
        docker_cmd = ["docker", "compose"]
        if shutil.which("docker-compose"):
            docker_cmd = ["docker-compose"]
            
        cmd = docker_cmd + ["restart", "model-server", "api"]
        log.info(f"Running command: {' '.join(cmd)}")
        
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True
        )
        for line in proc.stdout.splitlines():
            log.info(f"DOCKER: {line}")
            
        log.info("Containers restarted successfully!")

    except subprocess.CalledProcessError as e:
        log.error(f"Failed to restart containers. Return code: {e.returncode}")
        log.error(f"Stderr: {e.stderr}")
        return

    log.info("Deployment script completed successfully.")

if __name__ == "__main__":
    main()
