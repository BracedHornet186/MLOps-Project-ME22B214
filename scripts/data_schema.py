from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclasses.dataclass
class DataSchema:
    df: pd.DataFrame = dataclasses.field(repr=False)
    data_root_dir: str | Path = dataclasses.field()
    columns: tuple[str, ...] = dataclasses.field(init=False)

    def get_output_metadata(self, dataset: str, scene: str, name: str) -> dict:
        return {}

    def format_output_key(self, dataset: str, scene: str, name: str) -> str:
        raise NotImplementedError

    def build_image_relative_path(self, row: pd.Series) -> str:
        raise NotImplementedError

    def preprocess(self) -> None:
        pass

    def resolve_image_path(
        self, row: pd.Series, data_root_dir: Optional[str | Path] = None
    ) -> str:
        data_root_dir = Path(data_root_dir or self.data_root_dir)
        return str(data_root_dir / self.build_image_relative_path(row))

    def head(self, n: int) -> DataSchema:
        self.df = self.df.head(n=n)
        return self

    def apply(self, f: Callable) -> DataSchema:
        self.df = f(self.df)
        return self

    def check_files_exist(self, drop: bool = False, verbose: bool = True) -> DataSchema:
        is_found = self.df.apply(
            lambda row: Path(self.resolve_image_path(row)).exists(), axis=1
        )
        if is_found.sum() == len(self.df):
            return self

        missing_files = [
            self.resolve_image_path(self.df.iloc[i]) for i in self.df[~is_found].index
        ]
        missing_files = list(sorted(missing_files))
        for missing_file in missing_files:
            if verbose:
                print(f"Notfound: {missing_file}")

        if drop:
            self.df = self.df[is_found].reset_index(drop=True)
            if verbose:
                print("Missing files were removed from the dataframe")

        return self

    def get_image_paths(self) -> list[str]:
        return [self.resolve_image_path(self.df.iloc[i]) for i in range(len(self.df))]

    def get_scene_names(self) -> list[str]:
        return list(self.df["scene"].unique())

    @classmethod
    def create(
        cls,
        data_root_dir: str | Path,
        datasets_to_ignore: list[str] | None = None,
        datasets_to_use: list[str] | None = None,
        scenes_to_ignore: list[str] | None = None,
        scenes_to_use: list[str] | None = None,
    ) -> DataSchema:
        raise NotImplementedError
