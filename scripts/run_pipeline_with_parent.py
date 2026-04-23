import os
import subprocess
import mlflow

def main():
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "scene_reconstruction_dvc")

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="full_dvc_pipeline") as parent:
        parent_id = parent.info.run_id
        print(f"Parent run ID: {parent_id}")

        # Pass parent run to all stages
        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_id

        # Run DVC pipeline
        subprocess.run(["dvc", "repro"], check=True)
        
if __name__ == "__main__":
    main()