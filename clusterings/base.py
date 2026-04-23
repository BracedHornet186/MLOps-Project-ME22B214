from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np

from scripts.data import DEFAULT_OUTLIER_SCENE_NAME
from scripts.data_schema import DataSchema
from pipelines.scene import Scene

T = TypeVar("T")


class Clustering:
    def run(
        self,
        image_paths: list[str | Path],
        image_reader: Callable | None = None,
    ) -> ClusteringResult:
        raise NotImplementedError


@dataclass
class ClusteringResult:
    image_paths: list[str | Path]
    cluster_labels: np.ndarray

    additional_outputs: dict[str, Any] | None = None

    def get_output(self, key: str) -> Any | None:
        if self.additional_outputs is None:
            return None
        return self.additional_outputs.get(key)

    def add_output(self, key: str, value: Any) -> ClusteringResult:
        if self.additional_outputs is None:
            self.additional_outputs = {}
        if key in self.additional_outputs:
            print(f"Warning! ClusteringResult overwrites additional_outputs[{key}]")
        self.additional_outputs[key] = value
        return self

    def to_scene_list(self, dataset: str, data_schema: DataSchema) -> list[Scene]:
        image_dirs = [str(Path(path).parent) for path in self.image_paths]
        # All images from the same scene must be in the same directory
        assert len(set(image_dirs)) == 1
        image_dir = image_dirs[0]

        scenes = []
        cluster_groups = defaultdict(list)
        for i, cluster_id in enumerate(self.cluster_labels):
            cluster_groups[cluster_id].append(i)

        outlier_scene = None
        for cluster_id, indices in cluster_groups.items():
            image_paths = [self.image_paths[i] for i in indices]
            if cluster_id < 0:
                # Noisy label
                outlier_scene = Scene(
                    dataset=dataset,
                    scene=DEFAULT_OUTLIER_SCENE_NAME,
                    image_paths=image_paths,
                    image_dir=image_dir,
                    data_schema=data_schema,
                    indices_in_parent_scene=np.array(indices),
                )
            else:
                scene = Scene(
                    dataset=dataset,
                    scene=f"cluster{cluster_id}",
                    image_paths=image_paths,
                    image_dir=image_dir,
                    data_schema=data_schema,
                    indices_in_parent_scene=np.array(indices),
                )
                scenes.append(scene)

        if outlier_scene is not None:
            scenes.append(outlier_scene)

        print(f"[ClusteringResult] {len(scenes)} clusters have been created")
        for scene in scenes:
            print(f"  - {scene.scene}: {len(scene.image_paths)}")

        return scenes
