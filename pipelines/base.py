from typing import Optional
import pandas as pd
import tqdm

from scripts.data_schema import DataSchema
from pipelines.scene import Scene


class Pipeline:
    pipeline_id: Optional[str]

    def run(
        self, df: pd.DataFrame, data_schema: DataSchema, save_snapshot: bool = False
    ) -> pd.DataFrame:
        raise NotImplementedError

    def run_scene(
        self, scene: Scene, iterator: tqdm.tqdm, save_snapshot: bool = False
    ) -> dict:
        raise NotImplementedError

    def set_id(self, pipeline_id: Optional[str]):
        self.pipeline_id = pipeline_id
