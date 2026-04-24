from typing import List, Optional, Tuple

import tqdm

from scripts.data import FilePath
from pipelines.scene import Scene
from shortlists.base import ShortlistGenerator, get_all_pairs


class AllPairsShortlistGenerator(ShortlistGenerator):
    def __init__(self):
        pass

    def __call__(self,
                 scene: Scene,
                 progress_bar: Optional[tqdm.tqdm] = None,
                 **kwargs) -> List[Tuple[int, int]]:
        pairs = get_all_pairs(scene.image_paths)
        scene.update_shortlist(pairs)
        return pairs