import pandas as pd
import numpy as np
import pathlib
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys; sys.path.insert(0, "/app")
from data import IMC2025TrainData, DEFAULT_DATASET_DIR

def validate_data():
    schema = IMC2025TrainData.create(DEFAULT_DATASET_DIR)
    schema.preprocess()
    df = schema.df

    # 1. No duplicate image paths
    dupes = df.duplicated(subset=["image"]).sum()
    assert dupes == 0, f"{dupes} duplicate images"

    # 2. Rotation matrices are valid (9 values each)
    bad_R = df["rotation_matrix"].apply(
        lambda s: len(s.split(";")) != 9).sum()
    assert bad_R == 0, f"{bad_R} malformed rotation matrices"

    # 3. Translation vectors are valid (3 values each)
    bad_t = df["translation_vector"].apply(
        lambda s: len(s.split(";")) != 3).sum()
    assert bad_t == 0, f"{bad_t} malformed translation vectors"

    # 4. Scene coverage — at least 5 images per scene
    counts = df.groupby(["dataset","scene"]).size()
    small = (counts < 5).sum()
    print(f"[warn] {small} scenes with < 5 images")

    print(f"[OK] {len(df)} rows, {counts.shape[0]} scenes")

def check_custom_data():
    custom = pathlib.Path("data/train/custom_warehouse")
    images = list(custom.rglob("*.jpg")) + list(custom.rglob("*.png"))
    assert len(images) > 0, "No custom images found"
    print(f"[custom] {len(images)} images found")

with DAG("scene_reconstruction_ingest",
         start_date=datetime(2025, 1, 1),
         schedule_interval="@daily", catchup=False) as dag:

    t1 = PythonOperator(task_id="validate_imc25",
                        python_callable=validate_data)
    t2 = PythonOperator(task_id="validate_custom",
                        python_callable=check_custom_data)
    t1 >> t2