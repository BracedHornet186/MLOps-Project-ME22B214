from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn
import yaml

from scripts.config import SubmissionConfig
from scripts.data_schema import DataSchema


def _strtobool(val: str) -> int:
    """Replacement for deprecated distutils.util.strtobool."""
    val = val.strip().lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif val in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError(f"invalid truth value {val!r}")


# Const
# -------------------------------------
DEFAULT_DATASET_DIR = Path(os.environ.get("DEFAULT_DATASET_DIR", "data"))
DEFAULT_EXTRA_DIR = Path(os.environ.get("DEFAULT_EXTRA_DIR", "extra"))
DEFAULT_TMP_DIR = DEFAULT_EXTRA_DIR / "tmp"
DEFAULT_EVALUATION_ROOT_DIR = DEFAULT_EXTRA_DIR / "evaluations"
DEFAULT_SNAPSHOT_DIR = DEFAULT_EXTRA_DIR / "snapshot"
DEFAULT_SEED_VALUE = int(os.environ.get("DEFAULT_SEED_VALUE", 2025))
DEFAULT_SPACE_NAME = os.environ.get("DEFAULT_SPACE_NAME", str(int(time.time())))
DEFAULT_MODEL_LIST_PATH = Path(
    os.environ.get("DEFAULT_MODEL_LIST_PATH", "conf/models.yaml")
)
DEFAULT_ENV_NAME = os.environ.get("DEFAULT_ENV_NAME", None)

DEFAULT_OUTLIER_SCENE_NAME = "outliers"

DISTRIBUTED_SPLIT_OFFSET = int(os.environ.get("DISTRIBUTED_SPLIT_OFFSET", 0))


# Flags
# -------------------------------------
IS_SCENE_SPACE_DIR_PERSISTENT = _strtobool(
    os.environ.get("SCENE_SPACE_DIR_PERSISTENT", "no")
)
SAVE_CAMERA_DEBUG_INFO = _strtobool(
    os.environ.get("SAVE_CAMERA_DEBUG_INFO", "no")
)
SHOW_STATS = _strtobool(os.environ.get("SHOW_STATS", "no"))
SHOW_MATCHED_KEYPOINT_COUNT = _strtobool(
    os.environ.get("SHOW_MATCHED_KEYPOINT_COUNT", "no")
)
SHOW_PREF_TIME = _strtobool(os.environ.get("SHOW_PREF_TIME", "no"))
SHOW_MEM_USAGE = _strtobool(os.environ.get("SHOW_MEM_USAGE", "no"))

# Typing
# -------------------------------------
FilePath = Union[str, Path]
DirPath = Union[str, Path]

# (lafs, keypoints, scores, descriptors)
LocalFeatureExtractionOutputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

# (lafs, scores, descriptors)
LocalFeatureOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# Structures
# -------------------------------------
@dataclass
class Camera:
    rotmat: np.ndarray
    tvec: np.ndarray


# Vars
# -------------------------------------
ROTATION_THRESHOLDS_DEGREES_DICT = {
    **{
        ("haiper", scene): np.linspace(1, 10, 10)
        for scene in ["bike", "chairs", "fountain"]
    },
    **{("heritage", scene): np.linspace(1, 10, 10) for scene in ["cyprus", "dioscuri"]},
    **{("heritage", "wall"): np.linspace(0.2, 10, 10)},
    **{("urban", "kyiv-puppet-theater"): np.linspace(1, 10, 10)},
}  # Based on https://www.kaggle.com/code/eduardtrulls/imc2023-evaluation

TRANSLATION_THRESHOLDS_METERS_DICT = {
    **{
        ("haiper", scene): np.geomspace(0.05, 0.5, 10)
        for scene in ["bike", "chairs", "fountain"]
    },
    **{
        ("heritage", scene): np.geomspace(0.1, 2, 10)
        for scene in ["cyprus", "dioscuri"]
    },
    **{("heritage", "wall"): np.geomspace(0.05, 1, 10)},
    **{("urban", "kyiv-puppet-theater"): np.geomspace(0.5, 5, 10)},
}  # Based on https://www.kaggle.com/code/eduardtrulls/imc2023-evaluation


# Model paths
# -------------------------------------
if not DEFAULT_MODEL_LIST_PATH.exists():
    raise FileNotFoundError(DEFAULT_MODEL_LIST_PATH)

with open(DEFAULT_MODEL_LIST_PATH) as fp:
    MODEL_DEFS = yaml.safe_load(fp)


# Data schema
# -------------------------------------
class IMC2025TestData(DataSchema):
    columns = (
        "image_id",
        "dataset",
        "scene",
        "image",
        "rotation_matrix",
        "translation_vector",
    )

    def get_output_metadata(self, dataset: str, scene: str, name: str) -> dict:
        """image_id"""
        _df = self.df[self.df["image"] == name]
        assert len(_df) == 1
        image_id = str(_df.iloc[0]["image_id"])
        return {"image_id": image_id}

    def format_output_key(self, dataset: str, scene: str, name: str) -> str:
        """image"""
        return name

    def build_image_relative_path(self, row: pd.Series) -> str:
        return f"test/{row['dataset']}/{row['image']}"

    @classmethod
    def create(
        cls,
        data_root_dir: str | Path,
        datasets_to_ignore: list[str] | None = None,
        datasets_to_use: list[str] | None = None,
        scenes_to_ignore: list[str] | None = None,
        scenes_to_use: list[str] | None = None,
    ) -> DataSchema:
        return IMC2025TestData(load_submission_df(), data_root_dir=data_root_dir)


class IMC2025TrainData(DataSchema):
    columns = (
        "dataset",
        "scene",
        "image",
        "rotation_matrix",
        "translation_vector",
    )

    def format_output_key(self, dataset: str, scene: str, name: str) -> str:
        """image"""
        return name

    def build_image_relative_path(self, row: pd.Series) -> str:
        return f"train/{row['dataset']}/{row['image']}"

    def preprocess(self) -> None:
        self.check_files_exist(drop=True)

    @classmethod
    def create(
        cls,
        data_root_dir: str | Path,
        datasets_to_ignore: list[str] | None = None,
        datasets_to_use: list[str] | None = None,
        scenes_to_ignore: list[str] | None = None,
        scenes_to_use: list[str] | None = None,
    ) -> DataSchema:
        return IMC2025TrainData(
            load_train_df_imc2025(
                datasets_to_ignore=datasets_to_ignore,
                datasets_to_use=datasets_to_use,
                scenes_to_ignore=scenes_to_ignore,
                scenes_to_use=scenes_to_use,
            ),
            data_root_dir=data_root_dir,
        )


class IMC2024TestData(DataSchema):
    columns = (
        "image_path",
        "dataset",
        "scene",
        "rotation_matrix",
        "translation_vector",
    )

    def format_output_key(self, dataset: str, scene: str, name: str) -> str:
        return f"test/{scene}/images/{name}"

    def build_image_relative_path(self, row: pd.Series) -> str:
        return str(row["image_path"])

    @classmethod
    def create(
        cls,
        data_root_dir: str | Path,
        datasets_to_ignore: list[str] | None = None,
        datasets_to_use: list[str] | None = None,
        scenes_to_ignore: list[str] | None = None,
        scenes_to_use: list[str] | None = None,
    ) -> DataSchema:
        return IMC2024TestData(load_submission_df(), data_root_dir=data_root_dir)


class IMC2024TrainData(DataSchema):
    columns = (
        "image_name",
        "rotation_matrix",
        "translation_vector",
        "calibration_matrix",
        "dataset",
        "scene",
    )

    def format_output_key(self, dataset: str, scene: str, name: str) -> str:
        return name

    def build_image_relative_path(self, row: pd.Series) -> str:
        return f"train/{row['scene']}/images/{row['image_name']}"

    def preprocess(self) -> None:
        self.check_files_exist(drop=True)

    @classmethod
    def create(
        cls,
        data_root_dir: str | Path,
        datasets_to_ignore: list[str] | None = None,
        datasets_to_use: list[str] | None = None,
        scenes_to_ignore: list[str] | None = None,
        scenes_to_use: list[str] | None = None,
    ) -> DataSchema:
        return IMC2024TrainData(
            load_train_df(
                scenes_to_ignore=scenes_to_ignore, scenes_to_use=scenes_to_use
            ),
            data_root_dir=data_root_dir,
        )


class IMC2023TestData(DataSchema):
    columns = (
        "image_path",
        "dataset",
        "scene",
        "rotation_matrix",
        "translation_vector",
    )

    def format_output_key(self, dataset: str, scene: str, name: str) -> str:
        return f"{dataset}/{scene}/images/{name}"

    def build_image_relative_path(self, row: pd.Series) -> str:
        return row["image_path"]

    @classmethod
    def create(
        cls,
        data_root_dir: str | Path,
        datasets_to_ignore: list[str] | None = None,
        datasets_to_use: list[str] | None = None,
        scenes_to_ignore: list[str] | None = None,
        scenes_to_use: list[str] | None = None,
    ) -> DataSchema:
        return IMC2024TestData(load_submission_df(), data_root_dir=data_root_dir)


class IMC2023TrainData(DataSchema):
    columns = (
        "dataset",
        "scene",
        "image_path",
        "rotation_matrix",
        "translation_vector",
    )

    def format_output_key(self, dataset: str, scene: str, name: str) -> str:
        raise NotImplementedError

    def build_image_relative_path(self, row: pd.Series) -> str:
        return f"train/{row['image_path']}"

    @classmethod
    def create(
        cls,
        data_root_dir: str | Path,
        datasets_to_ignore: list[str] | None = None,
        datasets_to_use: list[str] | None = None,
        scenes_to_ignore: list[str] | None = None,
        scenes_to_use: list[str] | None = None,
    ) -> DataSchema:
        return IMC2023TrainData(
            load_train_df(
                scenes_to_ignore=scenes_to_ignore, scenes_to_use=scenes_to_use
            ),
            data_root_dir=data_root_dir,
        )


# Methods
# -------------------------------------
def set_random_seed(seed: int = DEFAULT_SEED_VALUE):
    print(f"SEED={seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def on_kaggle_kernel() -> bool:
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def on_kaggle_kernel_rerun() -> bool:
    return os.getenv("KAGGLE_IS_COMPETITION_RERUN") is not None


def resolve_model_path(
    key: str,
) -> FilePath:
    if Path(key).exists():
        return key
    if DEFAULT_ENV_NAME:
        return MODEL_DEFS[key][DEFAULT_ENV_NAME]
    if on_kaggle_kernel():
        return MODEL_DEFS[key]["kernel"]
    return MODEL_DEFS[key]["local"]


def load_train_df(
    replace_abs_path: bool = False,
    scenes_to_ignore: Optional[list[str]] = None,
    scenes_to_use: Optional[list[str]] = None,
) -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_DATASET_DIR / "train" / "train_labels.csv")
    if replace_abs_path:
        df = to_absolute_image_path(df, sub_dir="train")

    if scenes_to_use:
        df = df[df["scene"].isin(set(scenes_to_use))].reset_index(drop=True)
        print(f"Use scenes: {scenes_to_use}")

    if scenes_to_ignore:
        df = df[~df["scene"].isin(set(scenes_to_ignore))].reset_index(drop=True)
        print(f"Drop scenes: {scenes_to_ignore}")

    return df


def load_train_df_imc2025(
    datasets_to_ignore: Optional[list[str]] = None,
    datasets_to_use: Optional[list[str]] = None,
    scenes_to_ignore: Optional[list[str]] = None,
    scenes_to_use: Optional[list[str]] = None,
) -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_DATASET_DIR / "train_labels.csv")

    if datasets_to_use:
        df = df[df["dataset"].isin(set(datasets_to_use))].reset_index(drop=True)
        print(f"Use datasets: {datasets_to_use}")

    if datasets_to_ignore:
        df = df[~df["dataset"].isin(set(datasets_to_ignore))].reset_index(drop=True)
        print(f"Drop datasets: {datasets_to_ignore}")

    if scenes_to_use:
        df = df[df["scene"].isin(set(scenes_to_use))].reset_index(drop=True)
        print(f"Use scenes: {scenes_to_use}")

    if scenes_to_ignore:
        df = df[~df["scene"].isin(set(scenes_to_ignore))].reset_index(drop=True)
        print(f"Drop scenes: {scenes_to_ignore}")

    return df


def load_submission_df(replace_abs_path: bool = False) -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_DATASET_DIR / "sample_submission.csv")
    if replace_abs_path:
        df = to_absolute_image_path(df)
    return df


def to_absolute_image_path(df: pd.DataFrame, sub_dir: str = "") -> pd.DataFrame:
    df["image_path"] = df["image_path"].apply(
        lambda x: str(DEFAULT_DATASET_DIR / sub_dir / x)
    )
    return df


def arr_to_str(a: np.ndarray) -> str:
    return ";".join([str(x) for x in a.reshape(-1)])


def nan_R_str() -> str:
    return ";".join(["nan"] * 9)


def nan_t_str() -> str:
    return ";".join(["nan"] * 3)


def camera_dict_from_test_df(df: pd.DataFrame) -> dict:
    camera_dict = {}
    for i in range(len(df)):
        row = df.iloc[i]
        image_path = row["image_path"]
        dataset = row["dataset"]
        scene = row["scene"]
        R_str = row["rotation_matrix"]
        t_str = row["translation_vector"]
        R = np.fromstring(R_str.strip(), sep=";").reshape(3, 3)
        t = np.fromstring(t_str.strip(), sep=";")
        if dataset not in camera_dict:
            camera_dict[dataset] = {}
        if scene not in camera_dict[dataset]:
            camera_dict[dataset][scene] = {}
        camera_dict[dataset][scene][image_path] = Camera(rotmat=R, tvec=t)
    return camera_dict


def sample_image_path_imc2025(
    dataset: Optional[str] = None, scene: Optional[str] = None
) -> FilePath:
    datasets_to_use = None
    scenes_to_use = None
    if dataset:
        datasets_to_use = [dataset]
    if scene:
        scenes_to_use = [scene]
    df = load_train_df_imc2025(
        datasets_to_use=datasets_to_use,
        scenes_to_use=scenes_to_use,
    )
    row = df.sample(n=1).iloc[0]
    return str(DEFAULT_DATASET_DIR / "train" / row["dataset"] / row["image"])


def random_sample_image_pair_imc2025(
    data_schema: DataSchema, verbose: bool = False
) -> tuple[str, str]:
    df = data_schema.df
    target_dataset = random.choice(df["dataset"].values)
    pair_df = df[df["dataset"] == target_dataset].sample(n=2)
    path1 = data_schema.resolve_image_path(pair_df.iloc[0])
    path2 = data_schema.resolve_image_path(pair_df.iloc[1])
    pair = (path1, path2)
    if verbose:
        print(pair)
    return pair


def sample_image_path_imc2024(
    dataset: Optional[str] = None, scene: Optional[str] = None
) -> FilePath:
    scenes_to_use = None
    if dataset:
        raise NotImplementedError
    if scene:
        scenes_to_use = [scene]
    df = load_train_df(replace_abs_path=True, scenes_to_use=scenes_to_use)
    return str(df["image_path"].sample(n=1).values[0])


def random_sample_image_pair_imc2024(
    data_schema: DataSchema, verbose: bool = False
) -> tuple[str, str]:
    df = data_schema.df
    target_scene = random.choice(df["scene"].values)
    pair_df = df[df["scene"] == target_scene].sample(n=2)
    path1 = data_schema.resolve_image_path(pair_df.iloc[0])
    path2 = data_schema.resolve_image_path(pair_df.iloc[1])
    pair = (path1, path2)
    if verbose:
        print(pair)
    return pair


def setup_data_schema(
    conf: SubmissionConfig, data_root_dir: str | Path = DEFAULT_DATASET_DIR
) -> DataSchema:
    if conf.target_data_type in ("submission", "imc2025test"):
        data_schema = IMC2025TestData.create(data_root_dir=data_root_dir)
    elif conf.target_data_type in ("submission-fast-commit", "debug"):
        data_schema = IMC2025TestData.create(data_root_dir=data_root_dir).head(n=5)
    elif conf.target_data_type == "imc2024test":
        data_schema = IMC2024TestData.create(data_root_dir=data_root_dir)
    elif conf.target_data_type == "imc2023test":
        data_schema = IMC2023TestData.create(data_root_dir=data_root_dir)
    elif conf.target_data_type == "imc2025train":
        data_schema = IMC2025TrainData.create(
            data_root_dir=data_root_dir,
            datasets_to_ignore=conf.datasets_to_ignore,
            datasets_to_use=conf.datasets_to_use,
            scenes_to_ignore=conf.scenes_to_ignore,
            scenes_to_use=conf.scenes_to_use,
        )
    elif conf.target_data_type == "imc2024train":
        data_schema = IMC2024TrainData.create(
            data_root_dir=data_root_dir,
            scenes_to_ignore=conf.scenes_to_ignore,
            scenes_to_use=conf.scenes_to_use,
        )
    elif conf.target_data_type == "imc2023train":
        data_schema = IMC2023TrainData.create(
            data_root_dir=data_root_dir,
            scenes_to_ignore=conf.scenes_to_ignore,
            scenes_to_use=conf.scenes_to_use,
        )
    else:
        raise ValueError

    print(f"DataSchema: {data_schema}")
    data_schema.preprocess()
    return data_schema


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--split-version", default="v1")
    args = parser.parse_args()
