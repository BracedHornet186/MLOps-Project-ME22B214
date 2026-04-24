from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import tqdm

from matchers.base import DetectorFreeMatcher
from pipelines.scene import Scene
from pipelines.verification import compute_ransac_inlier_counts, run_ransac
from shortlists.base import ShortlistGenerator, ShortlistUpdater, get_all_pairs
from shortlists.config import (PreMatchingShortlistUpdaterConfig,
                               ShortlistGeneratorConfig)
from scripts.storage import InMemoryMatchedKeypointStorage
from scripts.workspace import log


class PreMatchingRandomWalkShortlistGenerator(ShortlistGenerator):
    def __init__(self, conf: ShortlistGeneratorConfig):
        self.conf = conf

    @torch.inference_mode()
    def __call__(self,
                 scene: Scene,
                 progress_bar: Optional[tqdm.tqdm] = None,
                 matcher: Optional[DetectorFreeMatcher] = None,
                 mk_storage: Optional[InMemoryMatchedKeypointStorage] = None,
                 **kwargs) -> List[Tuple[int, int]]:
        assert self.conf.random_walk_ransac is not None
        assert self.conf.random_walk_matching_threshold is not None
        assert self.conf.random_walk_num_trials_per_sample is not None
        assert self.conf.random_walk_num_references_per_sample is not None
        assert matcher is not None
        assert mk_storage is not None

        classname = self.__class__.__name__

        image_paths = scene.image_paths
        if len(image_paths) <= self.conf.global_desc_fallback_threshold:
            # Fallback to all_pairs
            log(f'# of images is less than '
                f'{self.conf.global_desc_fallback_threshold}')
            log(f'-> Use all pairs')

            pairs = get_all_pairs(image_paths)
            scene.update_shortlist(pairs)
            return pairs
        
        rng = np.random.default_rng(seed=1234)
        connection_counts = np.zeros((len(scene.image_paths),)).astype(np.int32)
        pairs = set()
        for i, path1 in enumerate(scene.image_paths):
            print(connection_counts)
            for trial in range(self.conf.random_walk_num_trials_per_sample):
                if connection_counts[i] >= self.conf.random_walk_num_references_per_sample:
                    break

                if trial == self.conf.random_walk_num_trials_per_sample:
                    print(f'[{classname}] {path1} reached {trial} trials')
                    break

                # Random sample
                j = rng.choice(
                    np.array([
                        a for a in range(len(scene.image_paths))
                        if a != i
                    ])
                )
                path2 = scene.image_paths[j]

                q = min(i, j)
                r = max(i, j)
                pair = (q, r)
                if pair in pairs:
                    print(f'[{classname}] pair({q}, {r}) has been already added. skip')
                    continue

                # Matching
                matcher(path1, path2, mk_storage, image_reader=scene.get_image)
                
                # Check
                mkpts1, mkpts2 = mk_storage.get(path1, path2)
                F, inliers = run_ransac(mkpts1, mkpts2, self.conf.random_walk_ransac)
                print(sum(inliers))

                if sum(inliers) < self.conf.random_walk_matching_threshold:
                    continue
                
                pairs.add(pair)
                connection_counts[i] += 1
                connection_counts[j] += 1
            
            if progress_bar:
                progress_bar.set_postfix_str(
                    f'{classname} ({i + 1}/{len(scene.image_paths)})'
                )
        return list(pairs)
