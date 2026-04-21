from __future__ import annotations

import os
import sys

import mlflow

ROOT_RUN_ID = sys.argv[1] if len(sys.argv) > 1 else None


def main() -> None:
    if not ROOT_RUN_ID:
        print("Usage: end_parent_dvc_run.py <run_id>", file=sys.stderr)
        sys.exit(1)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.MlflowClient().set_terminated(ROOT_RUN_ID, "FINISHED")
    print(f"Terminated MLflow parent run: {ROOT_RUN_ID}")


if __name__ == "__main__":
    main()
