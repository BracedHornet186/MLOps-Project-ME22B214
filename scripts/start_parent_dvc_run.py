from __future__ import annotations

import os

import mlflow


def main() -> None:
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("scene_reconstruction_dvc")

    with mlflow.start_run(run_name="full_dvc_pipeline") as run:
        print(run.info.run_id)


if __name__ == "__main__":
    main()
