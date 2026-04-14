from __future__ import annotations

import functools
from typing import Literal, Optional

import pandas as pd
import torch

from config import PipelineConfig, SubmissionConfig
from data import DEFAULT_SEED_VALUE, DataSchema, setup_data_schema
from extractor import LocalFeatureExtractor, extract_all
from features.config import LightGlueALIKEDConfig, LocalFeatureConfig
from features.factory import create_local_feature_handler
from pipeline import Pipeline, create_pipeline
from pipelines.common import create_data_dict, iterate_scenes
from pipelines.scene import Scene
from preprocesses.config import ResizeConfig

SCENE_SHORT_NAME_MAPPINGS = {
    "mttb": "multi-temporal-temple-baalshamin",
    "cup": "transp_obj_glass_cup",
    "cylinder": "transp_obj_glass_cylinder",
}


def reduce_samples(
    df: pd.DataFrame,
    max_num_samples: int | None = None,
    random_state: int = DEFAULT_SEED_VALUE,
    head: bool = False,
    filenames: Optional[list] = None,
    target_data_type: Literal["imc2024train", "imc2025train"] = "imc2025train",
) -> pd.DataFrame:
    if target_data_type == "imc2024train":
        group_by_label = "scene"
    elif target_data_type == "imc2025train":
        group_by_label = "dataset"
    else:
        raise ValueError

    filesnames = filenames or []
    dfs = []
    for _, idx in df.groupby(group_by_label).groups.items():
        if filenames:
            print(f"[reduce_samples] files={filenames}")
            dfs.append(df[df["image_name"].isin(set(filesnames))])
        elif max_num_samples and len(idx) > max_num_samples:
            if head:
                print(f"[reduce_samples] head={max_num_samples}")
                dfs.append(df.loc[idx].head(n=max_num_samples))
            else:
                dfs.append(
                    df.loc[idx].sample(n=max_num_samples, random_state=random_state)
                )
        else:
            dfs.append(df.loc[idx])
    df = pd.concat(dfs).reset_index(drop=True)
    return df


def setup_data_and_pipeline(
    config_path: str,
    dataset_names: list[str] | None = None,
    scene_names: list[str] | None = None,
    max_num_samples: int | None = None,
    head: bool = False,
    filenames: Optional[list] = None,
    target_data_type: Literal["imc2024train", "imc2025train"] = "imc2025train",
) -> tuple[DataSchema, Pipeline]:
    conf = SubmissionConfig(
        pipeline=PipelineConfig.load_config(config_path),
        target_data_type=target_data_type,
    )

    if target_data_type == "imc2024train" and scene_names:
        conf.scenes_to_use = scene_names
        conf.scenes_to_use = [
            SCENE_SHORT_NAME_MAPPINGS.get(s, s) for s in conf.scenes_to_use
        ]

    if target_data_type == "imc2025train" and dataset_names:
        conf.datasets_to_use = dataset_names

    data = setup_data_schema(conf)
    if max_num_samples or filenames:
        reducer = functools.partial(
            reduce_samples,
            max_num_samples=max_num_samples,
            head=head,
            filenames=filenames,
            target_data_type=target_data_type,
        )
        data = data.check_files_exist().apply(reducer)
    else:
        data = data.check_files_exist().apply(reduce_samples)

    pipeline = create_pipeline(conf.pipeline, device=torch.device("cuda:0"))
    return data, pipeline


def get_first_scene(data: DataSchema, pipeline: Pipeline) -> Scene:
    data_dict = create_data_dict(data, df=data.df)
    scene = next(iter(iterate_scenes(data_dict, data)))
    return scene
