from typing import List, Optional, Tuple

import tqdm

from scripts.data import FilePath
from pipelines.scene import Scene


class ShortlistGenerator:
    def __call__(
        self,
        scene: Scene,
        progress_bar: Optional[tqdm.tqdm] = None,
        **kwargs
    ) -> List[Tuple[int, int]]:
        raise NotImplementedError


class ShortlistUpdater:
    def __call__(
        self,
        scene: Scene,
        progress_bar: Optional[tqdm.tqdm] = None,
        **kwargs
    ) -> List[Tuple[int, int]]:
        raise NotImplementedError


class DebugShortlistGenerator(ShortlistGenerator):
    def __init__(self, num_pairs: int = 3):
        self.num_pairs = num_pairs

    def __call__(self,
                 scene: Scene,
                 progress_bar: Optional[tqdm.tqdm] = None,
                 **kwargs) -> List[Tuple[int, int]]:
        pairs = get_all_pairs(scene.image_paths)[:self.num_pairs]
        scene.update_shortlist(pairs)
        return pairs


def get_all_pairs(
    image_paths: List[FilePath]
) -> List[Tuple[int, int]]:
    index_pairs = []
    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            index_pairs.append((i, j))
    return index_pairs
 