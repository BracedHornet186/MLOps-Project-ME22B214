import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

from data import DEFAULT_SNAPSHOT_DIR
from pipelines.scene import Scene
from storage import (
    InMemoryKeypointStorage,
    InMemoryLocalFeatureStorage,
    InMemoryMatchingStorage,
    InMemoryTwoViewGeometryStorage
)


class SceneSnapshot:
    def __init__(
        self,
        scene: Scene,
        keypoint_storage: InMemoryKeypointStorage | InMemoryLocalFeatureStorage,
        matching_storage: InMemoryMatchingStorage,
        two_view_geometry_storage: Optional[InMemoryTwoViewGeometryStorage] = None
    ):
        self.scene = scene
        self.keypoint_storage = keypoint_storage
        self.matching_storage = matching_storage
        self.two_view_geometry_storage = two_view_geometry_storage

    def __str__(self) -> str:
        return scene_to_name(self.scene)

    def save(
        self,
        snapshot_file: Optional[str | Path] = None,
        pipeline_id: Optional[str] = None,
        release_cached_images: bool = True
    ) -> Path:
        name = str(self)

        if snapshot_file:
            snapshot_file = Path(snapshot_file)
        else:
            if pipeline_id:
                snapshot_file = Path(
                    DEFAULT_SNAPSHOT_DIR / pipeline_id / (name + ".pickle")
                )
            else:
                snapshot_file = Path(DEFAULT_SNAPSHOT_DIR / (name + ".pickle"))

        snapshot_file.parent.mkdir(parents=True, exist_ok=True)

        if release_cached_images:
            self.scene.release_cached_images()

        with open(snapshot_file, "wb") as fp:
            pickle.dump(self, fp)

        return snapshot_file

    @classmethod
    def load(cls, snapshot_file: str | Path) -> "SceneSnapshot":
        with open(snapshot_file, "rb") as fp:
            snapshot = pickle.load(fp)
        return snapshot

    @classmethod
    def find_by_scene(
        cls, scene: Scene, snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR
    ) -> "SceneSnapshot":
        snapshot_file = Path(snapshot_dir) / f"{scene_to_name(scene)}.pickle"
        return cls.load(snapshot_file)


def scene_to_name(scene: Scene) -> str:
    return f"{scene.data_schema.__class__.__name__}-num{len(scene.image_paths)}-{scene.dataset}-{scene.scene}"


def find_snapshots(
    snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
) -> dict[str, list[Path]]:
    snapshot_dir = Path(snapshot_dir)
    snapshot_files = list(sorted(snapshot_dir.glob("**/*")))
    snapshot_files = [f for f in snapshot_files if f.is_file()]

    snapshot_dict = defaultdict(list)
    for snapshot_file in snapshot_files:
        pipeline_id = snapshot_file.parent.name
        snapshot_dict[pipeline_id].append(snapshot_file)

    for key in snapshot_dict.keys():
        snapshot_dict[key] = list(sorted(snapshot_dict[key]))

    return snapshot_dict
