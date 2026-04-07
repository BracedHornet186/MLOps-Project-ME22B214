from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data import FilePath
from pipelines.scene import Scene
from postprocesses.base import MatchingFilter
from storage import InMemoryKeypointStorage, InMemoryMatchingStorage


class FeatureTrackMatchingFilter(MatchingFilter):
    def __init__(self) -> None:
        super().__init__()

    def run(
        self,
        keypoint_storage: InMemoryKeypointStorage,
        matching_storage: InMemoryMatchingStorage,
        scene: Scene,
        progress_bar: tqdm | None = None,
    ):
        clean_keypoints_and_matchings_based_on_feature_track(
            keypoint_storage, matching_storage
        )


@dataclasses.dataclass
class FeatureTrackInfo:
    key: str
    counts: np.ndarray
    connected_node_keys: list[str]

    @property
    def num_keypoints(self) -> int:
        return len(self.counts)

    def get_many_views_connected_track_ratio(self) -> int:
        # NOTE: (counts == 1) means two-view track only
        return (self.counts > 1).sum() / self.counts.sum()

    def get_two_view_connected_track_ratio(self) -> int:
        return (self.counts == 1).sum() / self.counts.sum()

    def get_many_views_connected_track_point_ratio(self) -> int:
        # NOTE: (counts == 1) means two-view track only
        return (self.counts > 1).sum() / self.num_keypoints

    def get_two_view_connected_track_point_ratio(self) -> int:
        return (self.counts == 1).sum() / self.num_keypoints


def analyze_feature_track_on_image(
    keypoint_storage: InMemoryKeypointStorage,
    matching_storage: InMemoryMatchingStorage,
    query_key: FilePath,
) -> FeatureTrackInfo:
    query_key = Path(query_key).name
    track_counts = np.zeros((len(keypoint_storage.get(query_key)),), dtype=np.int64)
    connected_node_keys = {query_key}
    for key1, group in matching_storage:
        for key2, idxs in group.items():
            if key1 != query_key and key2 != query_key:
                continue
            if not matching_storage.has(key1, key2):
                continue
            idxs = matching_storage.matches[key1][key2]
            if key1 == query_key:
                keypoint_ids = idxs[:, 0]
            elif key2 == query_key:
                keypoint_ids = idxs[:, 1]
            else:
                keypoint_ids = []
                print("!!!")

            if len(keypoint_ids) > 0:
                connected_node_keys.add(key1)
                connected_node_keys.add(key2)

            for keypoint_id in keypoint_ids:
                track_counts[keypoint_id] += 1

    connected_node_keys.remove(query_key)

    _connected_node_keys = list(sorted(list(connected_node_keys)))

    return FeatureTrackInfo(
        key=query_key, counts=track_counts, connected_node_keys=_connected_node_keys
    )


def clean_keypoints_and_matchings_based_on_feature_track(
    keypoint_storage: InMemoryKeypointStorage,
    matching_storage: InMemoryMatchingStorage,
) -> tuple[InMemoryKeypointStorage, InMemoryMatchingStorage]:
    for key1 in keypoint_storage.keypoints.keys():
        feature_track_info = analyze_feature_track_on_image(
            keypoint_storage, matching_storage, query_key=key1
        )
        if key1 not in matching_storage.matches:
            continue

        useless_keypoint_ids = set(np.where(feature_track_info.counts <= 2)[0])
        for key2 in matching_storage.matches[key1].keys():
            idxs = matching_storage.get(key1, key2)
            new_idxs = np.array(
                [pair for pair in idxs if pair[0] not in useless_keypoint_ids],
                dtype=idxs.dtype,
            )
            print(
                f"[clean_keypoints_and_matchings_based_on_feature_track]"
                f"({key1}, {key2}): {len(idxs)} -> {len(new_idxs)} matches"
            )
            if len(new_idxs) == 0:
                # Remove?
                new_idxs = np.empty((0, 2), dtype=idxs.dtype)
            matching_storage.add(key1, key2, new_idxs)
        print("---")

    return keypoint_storage, matching_storage
