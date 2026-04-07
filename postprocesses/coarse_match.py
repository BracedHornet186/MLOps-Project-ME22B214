"""Adapted from https://github.com/zju3dv/DetectorFreeSfM"""

import math
from pathlib import Path
from typing import ChainMap, Optional

import numpy as np
from tqdm import tqdm

from pipelines.scene import Scene
from storage import (
    InMemoryKeypointStorage,
    InMemoryMatchingStorage,
    MatchedKeypointStorage,
)


def run_coarse_matching_postprocess(
    storage: MatchedKeypointStorage,
    scene: Scene,
    match_round_ratio: int,
    pair_name_split: str = " ",
    keypoint_storage: Optional[InMemoryKeypointStorage] = None,  # out
    matching_storage: Optional[InMemoryMatchingStorage] = None,  # out
) -> tuple[InMemoryKeypointStorage, InMemoryMatchingStorage]:
    keypoint_storage = keypoint_storage or InMemoryKeypointStorage()
    matching_storage = matching_storage or InMemoryMatchingStorage()

    matches = to_detector_free_sfm_matches_format(storage)
    image_lists = [Path(path).name for path in scene.image_paths]
    #image_lists = [key for key, _ in storage]
    n_imgs = len(image_lists)

    print("[run_coarse_matching_postprocess] Combine keypoints!")
    all_kpts = Match2Kpts(matches, image_lists, name_split=pair_name_split)
    sub_kpts = [all_kpts]
    obj_refs = [keypoint_worker(sub_kpt) for sub_kpt in sub_kpts]
    keypoints = dict(ChainMap(*obj_refs))

    # Convert keypoints match to keypoints indexs
    print("[run_coarse_matching_postprocess] Update matches")
    obj_refs = [
        update_matches(
            sub_matches,
            keypoints,
            merge=True if match_round_ratio == 1 else False,
            pair_name_split=pair_name_split,
        )
        for sub_matches in split_dict(matches, math.ceil(len(matches) / 1))
    ]
    updated_matches = dict(ChainMap(*obj_refs))

    # Post process keypoints:
    keypoints = {k: v for k, v in keypoints.items() if isinstance(v, dict)}
    print("[run_coarse_matching_postprocess] Post-processing keypoints...")
    kpts_scores = [
        transform_keypoints(sub_kpts)
        for sub_kpts in split_dict(keypoints, math.ceil(len(keypoints) / 1))
    ]
    final_keypoints = dict(ChainMap(*[k for k, _ in kpts_scores]))
    final_scores = dict(ChainMap(*[s for _, s in kpts_scores]))

    # Reformat keypoints_dict and matches_dict
    # from (abs_img_path0 abs_img_path1) -> (img_name0, img_name1)
    for key, value in final_keypoints.items():
        keypoint_storage.add(key, value)

    for key, value in updated_matches.items():
        key1, key2 = key.split(pair_name_split)
        matching_storage.add(key1, key2, value)

    return keypoint_storage, matching_storage


def to_detector_free_sfm_matches_format(
    storage: MatchedKeypointStorage, pair_name_split: str = " "
) -> dict[str, np.ndarray]:
    storage = storage.to_memory()
    matches = {}
    for key1, group in storage:
        for key2, (mkpts1, mkpts2) in group.items():
            scores = storage.get_scores(key1, key2)
            assert scores is not None
            new_key = pair_name_split.join([key1, key2])
            new_match = np.concatenate(
                [mkpts1, mkpts2, scores[:, None]], axis=-1
            )  # (N, 5)
            matches[new_key] = new_match
    return matches


def split_dict(_dict, n):
    for _items in chunks(list(_dict.items()), n):
        yield dict(_items)


def chunks(lst, n, length=None):
    """Yield successive n-sized chunks from lst."""
    try:
        _len = len(lst)
    except TypeError as _:
        assert length is not None
        _len = length

    for i in range(0, _len, n):
        yield lst[i : i + n]
    # TODO: Check that lst is fully iterated


def agg_groupby_2d(keys, vals, agg="avg"):
    """
    Args:
        keys: (N, 2) 2d keys
        vals: (N,) values to average over
        agg: aggregation method
    Returns:
        dict: {key: agg_val}
    """
    assert agg in ["avg", "sum"]
    unique_keys, group, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    group_sums = np.bincount(group, weights=vals)
    values = group_sums if agg == "sum" else group_sums / counts
    return dict(zip(map(tuple, unique_keys), values))


class Match2Kpts(object):
    """extract all possible keypoints for each image from all image-pair matches"""

    def __init__(self, matches, names, name_split="-", cov_threshold=0):
        self.names = names
        self.matches = matches
        self.cov_threshold = cov_threshold
        self.name2matches = {name: [] for name in names}
        for k in matches.keys():
            try:
                name0, name1 = k.split(name_split)
            except ValueError as _:
                name0, name1 = k.split("-")
            self.name2matches[name0].append((k, 0))
            self.name2matches[name1].append((k, 1))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            name = self.names[idx]
            if len(self.name2matches[name]) == 0:
                return name, np.empty((0, 3))

            kpts = np.concatenate(
                [
                    self.matches[k][:, [2 * id, 2 * id + 1, 4]]
                    for k, id in self.name2matches[name]
                    if self.matches[k].shape[0] >= self.cov_threshold
                ],
                0,
            )
            return name, kpts
        elif isinstance(idx, slice):
            names = self.names[idx]
            try:
                kpts = [
                    np.concatenate(
                        [
                            self.matches[k][:, [2 * id, 2 * id + 1, 4]]
                            for k, id in self.name2matches[name]
                            if self.matches[k].shape[0] >= self.cov_threshold
                        ],
                        0,
                    )
                    for name in names
                ]
            except:
                kpts = []
                for name in names:
                    kpt = [
                        self.matches[k][:, [2 * id, 2 * id + 1, 4]]
                        for k, id in self.name2matches[name]
                        if self.matches[k].shape[0] >= self.cov_threshold
                    ]
                    if len(kpt) != 0:
                        kpts.append(np.concatenate(kpt, 0))
                    else:
                        kpts.append(np.empty((0, 3)))
                        print(f"no keypoints in image:{name}")
            return list(zip(names, kpts))
        else:
            raise TypeError(f"{type(self).__name__} indices must be integers")


def keypoint_worker(name_kpts, pba=None, verbose=True):
    """merge keypoints associated with one image."""
    keypoints = {}

    if verbose:
        name_kpts = tqdm(name_kpts) if pba is None else name_kpts
    else:
        assert pba is None

    for name, kpts in name_kpts:
        kpt2score = agg_groupby_2d(kpts[:, :2].astype(int), kpts[:, -1], agg="sum")
        kpt2id_score = {
            k: (i, v)
            for i, (k, v) in enumerate(
                sorted(kpt2score.items(), key=lambda kv: kv[1], reverse=True)
            )
        }
        keypoints[name] = kpt2id_score

        if pba is not None:
            pba.update.remote(1)
    return keypoints


def update_matches(matches, keypoints, merge=False, pba=None, verbose=True, **kwargs):
    # convert match to indices
    ret_matches = {}

    if verbose:
        matches_items = tqdm(matches.items()) if pba is None else matches.items()
    else:
        assert pba is None
        matches_items = matches.items()

    for k, v in matches_items:
        mkpts0, mkpts1 = (
            map(tuple, v[:, :2].astype(int)),
            map(tuple, v[:, 2:4].astype(int)),
        )
        name0, name1 = k.split(kwargs["pair_name_split"])
        _kpts0, _kpts1 = keypoints[name0], keypoints[name1]

        mids = np.array(
            [
                [_kpts0[p0][0], _kpts1[p1][0]]
                for p0, p1 in zip(mkpts0, mkpts1)
                if p0 in _kpts0 and p1 in _kpts1
            ]
        )

        if len(mids) == 0:
            mids = np.empty((0, 2))

        def _merge_possible(name):  # only merge after dynamic nms (for now)
            return f"{name}_no-merge" not in keypoints

        if merge and _merge_possible(name0) and _merge_possible(name1):
            merge_ids = []
            mkpts0, mkpts1 = (
                map(tuple, v[:, :2].astype(int)),
                map(tuple, v[:, 2:4].astype(int)),
            )
            for p0, p1 in zip(mkpts0, mkpts1):
                if (*p0, -2) in _kpts0 and (*p1, -2) in _kpts1:
                    merge_ids.append([_kpts0[(*p0, -2)][0], _kpts1[(*p1, -2)][0]])
                elif p0 in _kpts0 and (*p1, -2) in _kpts1:
                    merge_ids.append([_kpts0[p0][0], _kpts1[(*p1, -2)][0]])
                elif (*p0, -2) in _kpts0 and p1 in _kpts1:
                    merge_ids.append([_kpts0[(*p0, -2)][0], _kpts1[p1][0]])
            merge_ids = np.array(merge_ids)

            if len(merge_ids) == 0:
                merge_ids = np.empty((0, 2))
                print("merge failed! No matches have been merged!")
            else:
                print(f"merge successful! Merge {len(merge_ids)} matches")

            try:
                mids_multiview = np.concatenate([mids, merge_ids], axis=0)
            except ValueError:
                # ?
                pass

            mids = np.unique(mids_multiview, axis=0)
        else:
            assert (
                len(mids) == v.shape[0]
            ), f"len mids: {len(mids)}, num matches: {v.shape[0]}"

        ret_matches[k] = mids.astype(int)  # (N,2)
        if pba is not None:
            pba.update.remote(1)

    return ret_matches


def transform_keypoints(keypoints, pba=None, verbose=True):
    """assume keypoints sorted w.r.t. score"""
    ret_kpts = {}
    ret_scores = {}

    if verbose:
        keypoints_items = tqdm(keypoints.items()) if pba is None else keypoints.items()
    else:
        assert pba is None
        keypoints_items = keypoints.items()

    for k, v in keypoints_items:
        v = {_k: _v for _k, _v in v.items() if len(_k) == 2}
        kpts = np.array([list(kpt) for kpt in v.keys()]).astype(np.float32)
        scores = np.array([s[-1] for s in v.values()]).astype(np.float32)
        if len(kpts) == 0:
            print("corner-case n_kpts=0 exists!")
            kpts = np.empty((0, 2))
        ret_kpts[k] = kpts
        ret_scores[k] = scores
        if pba is not None:
            pba.update.remote(1)
    return ret_kpts, ret_scores
