from __future__ import annotations

import torch
import tqdm
from mast3r.retrieval.graph import make_pairs_fps

from pipelines.scene import Scene
from retrievers.factory import MASt3RRetrievalASMKRetriever
from shortlists.base import ShortlistGenerator, get_all_pairs
from shortlists.config import ShortlistGeneratorConfig


class MASt3RRetrievalASMKShortlistGenerator(ShortlistGenerator):
    def __init__(
        self, conf: ShortlistGeneratorConfig, device: torch.device | None = None
    ):
        self.conf = conf
        self.device = device
        assert conf.mast3r_retrieval_asmk
        self.retriever = MASt3RRetrievalASMKRetriever(
            conf.mast3r_retrieval_asmk, device=device
        )

    @torch.inference_mode()
    def __call__(
        self, scene: Scene, progress_bar: tqdm.tqdm | None = None, **kwargs
    ) -> list[tuple[int, int]]:
        image_paths = scene.image_paths
        if (
            self.conf.mast3r_retrieval_asmk_fallback_threshold is not None
            and len(image_paths) <= self.conf.mast3r_retrieval_asmk_fallback_threshold
        ):
            # Fallback to all_pairs
            print(
                f"[{self.__class__.__name__}] # of images is less than {self.conf.mast3r_retrieval_asmk_fallback_threshold}"
            )
            print(f"[{self.__class__.__name__}] -> Use all pairs")
            pairs = get_all_pairs(image_paths)
            scene.update_shortlist(pairs)
            return pairs

        print(f"[{self.__class__.__name__}] Creating a sim matrix")
        sims = self.retriever.build(image_paths).get_sim_matrix(image_paths)

        print(f"[{self.__class__.__name__}] Making pairs")
        fps_pairs, anchor_idxs = make_pairs_fps(
            sims,
            Na=self.conf.mast3r_retrieval_asmk_make_pairs_fps_n,
            tokK=self.conf.mast3r_retrieval_asmk_make_pairs_fps_k,
            dist_thresh=self.conf.mast3r_retrieval_asmk_make_pairs_fps_dist_threshold,
        )

        pairs = []
        for i, j in fps_pairs:
            if self.conf.mast3r_retrieval_asmk_remove_swapped_pairs:
                pair = tuple(sorted((int(i), int(j))))
            else:
                pair = (int(i), int(j))
            pairs.append(pair)

        pairs = sorted(list(set(pairs)))
        scene.update_shortlist(pairs)
        return pairs
