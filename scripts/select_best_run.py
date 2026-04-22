from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import mlflow


ROOT = Path(__file__).resolve().parent.parent


def _list_artifacts_recursive(
    client: mlflow.tracking.MlflowClient,
    run_id: str,
    path: str = "",
) -> list[str]:
    files: list[str] = []
    stack = [path]
    while stack:
        cur = stack.pop()
        for artifact in client.list_artifacts(run_id, path=cur):
            if artifact.is_dir:
                stack.append(artifact.path)
            else:
                files.append(artifact.path)
    return files


def _select_config_artifact_path(
    client: mlflow.tracking.MlflowClient,
    run_id: str,
) -> str:
    artifact_paths = _list_artifacts_recursive(client, run_id)
    if not artifact_paths:
        raise RuntimeError(f"No artifacts found for run {run_id}")

    yaml_artifacts = [p for p in artifact_paths if p.lower().endswith((".yaml", ".yml"))]
    if not yaml_artifacts:
        raise RuntimeError(
            f"No YAML config artifact found in run {run_id}. "
            f"Available artifacts: {artifact_paths}"
        )

    # Prefer explicit config.yaml first, then config/*, then any YAML artifact.
    for path in yaml_artifacts:
        if Path(path).name == "config.yaml":
            return path

    for path in yaml_artifacts:
        if path.startswith("config/"):
            return path

    return yaml_artifacts[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select best evaluate run by highest mAA_overall")
    parser.add_argument("--experiment-name", default="scene_reconstruction_dvc", help="MLflow experiment name (parent runs are searched)")
    parser.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking server URI",
    )
    parser.add_argument("--run-name", default="full_dvc_pipeline", help="Run name to consider for best-run selection")
    parser.add_argument(
        "--best-config-path",
        default=str(ROOT / "conf" / "best_config.yaml"),
        help="Local path where the selected run config artifact is saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_config_path = Path(args.best_config_path)

    mlflow.set_tracking_uri(args.mlflow_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=args.mlflow_uri)

    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment not found: {args.experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=50000,
    )

    candidates = []
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "")
        if run_name != args.run_name:
            continue
        mAA_overall = run.data.metrics.get("mAA_overall")
        if mAA_overall is None:
            continue
        start_time = int(run.info.start_time or 0)
        candidates.append((float(mAA_overall), start_time, run))

    if not candidates:
        raise RuntimeError(f"No finished '{args.run_name}' runs with metric 'mAA_overall' found")

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_mAA_overall, _best_start_time, best_run = candidates[0]

    best_run_id = best_run.info.run_id
    parent_run_id = best_run.data.tags.get("mlflow.parentRunId", "")

    client.set_tag(best_run_id, "stage", "production")

    artifact_path = _select_config_artifact_path(client, best_run_id)
    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="best_run_cfg_") as tmpdir:
        downloaded_path = client.download_artifacts(
            best_run_id,
            artifact_path,
            dst_path=tmpdir,
        )
        shutil.copyfile(downloaded_path, best_config_path)

    print(
        json.dumps(
            {
                "best_run_id": best_run_id,
                "best_mAA_overall": best_mAA_overall,
                "parent_run_id": parent_run_id,
                "config_artifact": artifact_path,
                "best_config_path": str(best_config_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
