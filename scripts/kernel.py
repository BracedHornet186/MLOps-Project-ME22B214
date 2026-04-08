from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch

from config import PipelineConfig, SubmissionConfig
from data import (
    DEFAULT_DATASET_DIR,
    DISTRIBUTED_SPLIT_OFFSET,
    IMC2023TestData,
    IMC2023TrainData,
    IMC2024TestData,
    IMC2024TrainData,
    load_submission_df,
    load_train_df,
    on_kaggle_kernel,
    on_kaggle_kernel_rerun,
    setup_data_schema,
)
from distributed import DistConfig, init_dist
from pipeline import create_pipeline
from workspace import log


def run(
    conf: SubmissionConfig,
    env_name: str = "kernel",
    data_root_dir: str | Path = DEFAULT_DATASET_DIR,
    dist_conf: Optional[DistConfig] = None,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    cv2.setNumThreads(1)

    dist_conf = dist_conf or DistConfig.single()
    device = dist_conf.device
    print(f"device = {device}")

    # Load data
    data_schema = setup_data_schema(conf, data_root_dir=data_root_dir)

    # Distributed setup
    if dist_conf.is_master():
        conf.dist_output_dir_path.mkdir(parents=True, exist_ok=True)

    if dist_conf and dist_conf.world_size > 1:
        # For IMC2025
        func = partial(split_df_by_dataset, dist_conf=dist_conf)
        data_schema = data_schema.apply(func)
        print(f"[rank={dist_conf.rank}] Datasets: {data_schema.df['dataset'].unique()}")

    pipeline = create_pipeline(conf.pipeline, device=device)
    submission_df = pipeline.run(data_schema.df, data_schema)
    return submission_df


def run_and_save_submission(
    conf: SubmissionConfig,
    env_name: str = "kernel",
    data_root_dir: str | Path = DEFAULT_DATASET_DIR,
    dist_conf: Optional[DistConfig] = None,
) -> None:
    submission_df = run(
        conf, env_name=env_name, dist_conf=dist_conf, data_root_dir=data_root_dir
    )
    if dist_conf:
        submission_df.to_csv(
            conf.dist_output_dir_path / f"submission_{dist_conf.rank}.csv", index=False
        )
        print(f"[rank={dist_conf.rank}] Saved: submission.csv")
        if dist_conf.is_master():
            submission_df = wait_all_submissions(conf, dist_conf)
            submission_df.to_csv("submission.csv", index=False)
        print(f"[rank={dist_conf.rank}] Done")
    else:
        submission_df.to_csv("submission.csv", index=False)


def split_df_by_dataset(df: pd.DataFrame, dist_conf: DistConfig) -> pd.DataFrame:
    all_datasets = sorted(list(df["dataset"].unique()))
    if DISTRIBUTED_SPLIT_OFFSET == 0:
        datasets = [
            dataset
            for i, dataset in enumerate(all_datasets)
            if i % dist_conf.world_size == dist_conf.rank
        ]
    else:
        print("****************************************************")
        print(f"DISTRIBUTED_SPLIT_OFFSET={DISTRIBUTED_SPLIT_OFFSET}")
        print("****************************************************")
        datasets = [
            dataset
            for i, dataset in enumerate(all_datasets, start=DISTRIBUTED_SPLIT_OFFSET)
            if i % dist_conf.world_size == dist_conf.rank
        ]
    df = df[df["dataset"].isin(set(datasets))].reset_index(drop=True)
    return df


def split_df_by_scene(df: pd.DataFrame, dist_conf: DistConfig) -> pd.DataFrame:
    all_scenes = sorted(list(df["scene"].unique()))
    scenes = [
        scene
        for i, scene in enumerate(all_scenes)
        if i % dist_conf.world_size == dist_conf.rank
    ]
    df = df[df["scene"].isin(set(scenes))].reset_index(drop=True)
    return df


def wait_all_submissions(conf: SubmissionConfig, dist_conf: DistConfig) -> pd.DataFrame:
    submission_part_files = []
    print(f"[rank={dist_conf.rank}] Waiting all submissions ...")
    while True:
        submission_part_files = list(conf.dist_output_dir_path.glob("*.csv"))
        if len(submission_part_files) == dist_conf.world_size:
            print("Found: all submission parts")
            break
        time.sleep(5)
        print(f"[rank={dist_conf.rank}] Waiting all submissions ...")

    part_dfs = [pd.read_csv(f) for f in submission_part_files]
    df = pd.concat(part_dfs, axis=0).reset_index(drop=True)

    for f in submission_part_files:
        f.unlink(missing_ok=True)
    return df


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-p", "--pipeline-conf", default=None, help="Path to pipeline config"
    )
    parser.add_argument(
        "-c", "--submission-conf", default=None, help="Path to pipeline config"
    )
    parser.add_argument("--env-name", default="local")
    parser.add_argument("--dist", action="store_true")
    parser.add_argument(
        "-d",
        "--target-data-type",
        default="submission",
        choices=(
            "submission",
            "submission-fast-commit",
            "debug",
            "imc2025test",
            "imc2025train",
            "imc2024test",
            "imc2024train",
            "imc2023test",
            "imc2023train",
        ),
    )
    parser.add_argument("--data-root-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--kaggle-submit", action="store_true")
    args = parser.parse_args()

    if args.pipeline_conf:
        conf = SubmissionConfig(
            pipeline=PipelineConfig.load_config(args.pipeline_conf),
            target_data_type=args.target_data_type,
        )
    elif args.submission_conf:
        conf = SubmissionConfig.load_config(args.submission_conf)
    else:
        raise RuntimeError

    if args.kaggle_submit:
        if not on_kaggle_kernel_rerun():
            conf.target_data_type = "submission-fast-commit"
        assert args.env_name == "kernel"
    else:
        if args.datasets:
            conf.datasets_to_use = args.datasets
        if args.scenes:
            conf.scenes_to_use = args.scenes

    if args.dist:
        log("Distributed: On")
        dist_conf = init_dist(ddp=True)
        torch.use_deterministic_algorithms(True)
    else:
        dist_conf = None

    run_and_save_submission(
        conf,
        env_name=args.env_name,
        data_root_dir=args.data_root_dir,
        dist_conf=dist_conf,
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
