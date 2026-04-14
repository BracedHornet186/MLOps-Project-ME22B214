from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import torch
import tqdm
import numpy as np

from pipelines.scene import Scene
from shortlists.base import ShortlistUpdater
from shortlists.config import LocalFeatureBasedANNShortlistUpdaterConfig
from storage import LocalFeatureStorage


class LocalFeatureBasedANNShortlistUpdater(ShortlistUpdater):
    def __init__(self, conf: LocalFeatureBasedANNShortlistUpdaterConfig):
        self.conf = conf
    
    def __call__(self, scene: Scene,
                 progress_bar: Optional[tqdm.tqdm] = None,
                 feature_storage: Optional[LocalFeatureStorage] = None,
                 **kwargs) -> List[Tuple[int, int]]:
        assert feature_storage
        f_storage = feature_storage.to_memory()

        num_gpu = torch.cuda.device_count()
        gpu = 0
        if num_gpu == 2:
            gpu = 1
        print(f'[{self.__class__.__name__}] Use GPU={gpu}')

        index = faiss.IndexFlatL2(self.conf.dim)
        faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), gpu, index)

        queries = []
        ids = []
        for i, path in enumerate(scene.image_paths):
            key = Path(path).name
            descs = f_storage.descriptors[key]
            scores = f_storage.scores[key]
            keeps = (-scores).argsort()[:self.conf.num_limit_index_feature]
            descs = descs[keeps]
            index.add(descs)    # type: ignore
            queries.append(descs[:self.conf.num_limit_query_feature])
            ids += [i] * len(descs)
            if progress_bar:
                progress_bar.set_postfix_str(
                    f'Indexing ({i + 1}/{len(scene.image_paths)})'
                )
        
        ids = np.array(ids, dtype=np.uint16)
        for i, path in enumerate(scene.image_paths):
            qs = queries[i]
            dists, ranks = index.search(qs, self.conf.k)   # type: ignore
            topk_ids = ids[ranks]
            topk_ids = topk_ids[topk_ids != i]      # Remove self
            unique, counts = np.unique(topk_ids, return_counts=True)
            freqs = np.array([unique, counts]).T
