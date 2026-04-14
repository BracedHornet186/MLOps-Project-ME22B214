from typing import Any, Optional

import numpy as np
import tqdm

from pipelines.scene import Scene
from storage import InMemoryKeypointStorage, InMemoryMatchingStorage


class Localizer:
    def localize(self, scene: Scene, pairs: list[tuple[int, int]]) -> dict:
        raise NotImplementedError


class PostLocalizer:
    def localize(
        self,
        reference_sfm: Any,
        no_registered_query_output_keys: list[str],
        outputs: dict[str, dict[str, np.ndarray]],
        scene: Scene,
        keypoint_storage: InMemoryKeypointStorage,
        matching_storage: InMemoryMatchingStorage,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> dict:
        raise NotImplementedError
