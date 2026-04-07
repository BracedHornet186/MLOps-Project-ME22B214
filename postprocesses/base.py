from __future__ import annotations

from typing import Optional

import tqdm

from pipelines.common import Scene
from storage import (
    InMemoryKeypointStorage,
    InMemoryMatchingStorage,
    InMemoryTwoViewGeometryStorage,
)


class MatchingFilter:
    def run(
        self,
        keypoint_storage: InMemoryKeypointStorage,
        matching_storage: InMemoryMatchingStorage,
        scene: Scene,
        progress_bar: Optional[tqdm.tqdm] = None,
    ):
        raise NotImplementedError


class TwoViewGeometryPruner:
    def __call__(
        self,
        scene: Scene,
        g_storage: InMemoryTwoViewGeometryStorage,
        keypoint_storage: InMemoryKeypointStorage | None = None,
        database_path: str = "colmap.db",
        progress_bar: Optional[tqdm.tqdm] = None,
    ):
        raise NotImplementedError
