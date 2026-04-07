from typing import Optional
from collections import defaultdict

import pandas as pd

from data import FilePath
from pipelines.base import Pipeline
from shortlists.base import get_all_pairs
from storage import (InMemoryKeypointStorage, InMemoryMatchingStorage,
                     InMemoryTwoViewGeometryStorage)


class PipelineVisualizer:
    def __init__(self, pipeline: Pipeline,
                 df: pd.DataFrame):
        self.pipeline = pipeline
        self.df = df
    
    def __call__(self, path1: FilePath, path2: FilePath):
        df = self.df[self.df['image_path'].isin(set([str(path1), str(path2)]))]
        result_df = self.pipeline.run(df)


class StorageVisualizer:
    def __init__(self,
                 keypoint_storage: InMemoryKeypointStorage,
                 matching_storage: InMemoryMatchingStorage,
                 geometry_storage: Optional[InMemoryTwoViewGeometryStorage] = None):
        self.k_storage = keypoint_storage
        self.m_storage = matching_storage
        self.g_storage = geometry_storage
        print(f'count={self.g_storage.count}')
    
    def show_stats(self):
        all_keys = []
        for key in self.k_storage.keypoints.keys():
            all_keys.append(key)
        pairs = get_all_pairs(all_keys)

        data = defaultdict(list)
        for i, j in pairs:
            key1, key2 = all_keys[i], all_keys[j]

            data['key1'].append(key1)
            data['key2'].append(key2)
            data['num_keypoints1'].append(len(self.k_storage.get(key1)))
            data['num_keypoints2'].append(len(self.k_storage.get(key2)))
            try:
                data['num_matches'].append(len(self.m_storage.matches[key1][key2]))
            except KeyError as e:
                data['num_matches'].append(None)

            if self.g_storage:
                try:
                    inliers = self.g_storage.inliers[key1][key2]
                    data['num_inliers'].append(len(inliers))
                except KeyError as e:
                    data['num_inliers'].append(None)
            else:
                data['num_inliers'].append(None)
            
        df = pd.DataFrame(data)
        df.to_csv('.stats.csv', index=False)