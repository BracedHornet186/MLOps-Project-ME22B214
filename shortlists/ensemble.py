from __future__ import annotations

from typing import Optional

import tqdm

from pipelines.scene import Scene
from shortlists.base import ShortlistGenerator
from shortlists.config import EnsembleShortlistGeneratorConfig
from shortlists.exhaustive import get_all_pairs


class EnsembleShortlistGenerator(ShortlistGenerator):
    def __init__(
        self,
        conf: EnsembleShortlistGeneratorConfig,
        shortlist_generators: list[ShortlistGenerator],
    ):
        self.conf = conf
        self.shortlist_generators = shortlist_generators
        print(
            f"EnsembleShortlistGenerator | shortlist_generators: "
            f"{[g.__class__.__name__ for g in self.shortlist_generators]}"
        )

    def __call__(
        self, scene: Scene, progress_bar: Optional[tqdm.tqdm] = None, **kwargs
    ) -> list[tuple[int, int]]:
        if len(scene.image_paths) <= self.conf.all_pairs_fallback_threshold:
            # Fallback to all_pairs
            print(f"# of images is less than {self.conf.all_pairs_fallback_threshold}")
            print("-> Use all pairs")

            pairs = get_all_pairs(scene.image_paths)
            topk_ranks, topk_dists = None, None
            scene.update_shortlist(pairs).update_topk_table(topk_ranks, topk_dists)
            return pairs

        pairs = []
        stats = []
        for shortlist_generator in self.shortlist_generators:
            _pairs = shortlist_generator(scene, progress_bar=progress_bar, **kwargs)
            pairs += _pairs
            stats += (shortlist_generator.__class__.__name__, len(_pairs), len(pairs))

        pairs = list(set(pairs))
        pairs = [(i, j) if i < j else (j, i) for i, j in pairs]
        pairs = sorted(list(set(pairs)))

        print("--------------------------")
        print("EnsembleShortlistGenerator")
        print("--------------------------")
        for stat in stats:
            print(stat)
        print("--------------------------")

        scene.update_shortlist(pairs)
        return pairs
