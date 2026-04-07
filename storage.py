from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from data import SHOW_MEM_USAGE, DirPath, FilePath, LocalFeatureExtractionOutputs
from pipelines.scene import Scene


class Storage:
    def add(self, *args, **kwargs):
        raise NotImplementedError

    def get(self, *args, **kwargs):
        raise NotImplementedError


class LocalFeatureStorage(Storage):
    def add(
        self,
        imagefile: FilePath,
        outputs: LocalFeatureExtractionOutputs,
    ) -> None:
        raise NotImplementedError

    def check_size(self) -> None:
        raise NotImplementedError

    def get(self, path: FilePath) -> LocalFeatureExtractionOutputs:
        raise NotImplementedError

    def to_memory(self) -> InMemoryLocalFeatureStorage:
        raise NotImplementedError


class KeypointStorage(Storage):
    def add(self, path: FilePath, kpts: np.ndarray) -> None:
        raise NotImplementedError

    def get(self, path: FilePath) -> np.ndarray:
        raise NotImplementedError

    def to_memory(self) -> InMemoryKeypointStorage:
        raise NotImplementedError


class MatchedKeypointStorage(Storage):
    def add(
        self,
        path1: FilePath,
        path2: FilePath,
        kpts1: np.ndarray,
        kpts2: np.ndarray,
        scores: Optional[np.ndarray] = None,
    ) -> None:
        raise NotImplementedError

    def __iter__(
        self,
    ) -> Iterator[tuple[str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
        raise NotImplementedError

    def get(self, path1: FilePath, path2: FilePath) -> np.ndarray:
        raise NotImplementedError

    def get_scores(self, path1: FilePath, path2: FilePath) -> Optional[np.ndarray]:
        raise NotImplementedError

    def has(self, path1: FilePath, path2: FilePath) -> bool:
        raise NotImplementedError

    def has_scores(self, path1: FilePath, path2: FilePath) -> bool:
        raise NotImplementedError

    def to_memory(self) -> InMemoryMatchedKeypointStorage:
        raise NotImplementedError

    def to_keypoints_and_matches(
        self,
        keypoint_storage: Optional[KeypointStorage] = None,
        matching_storage: Optional[MatchingStorage] = None,
        apply_round: bool = True,
    ) -> tuple[KeypointStorage, MatchingStorage]:
        return convert_matched_keypoints_to_keypoints(
            self,
            keypoint_storage=keypoint_storage,
            matching_storage=matching_storage,
            apply_round=apply_round,
        )

    def from_keypoints_and_matches(
        self, keypoint_storage: KeypointStorage, matching_storage: MatchingStorage
    ) -> MatchedKeypointStorage:
        for key1, group in matching_storage:
            for key2, idxs in group.items():
                if self.has(key1, key2):
                    raise RuntimeError(f"Already exists: ({key1}, {key2})")
                kpts1 = keypoint_storage.get(key1)
                kpts2 = keypoint_storage.get(key2)
                mkpts1 = kpts1[idxs[:, 0]]
                mkpts2 = kpts2[idxs[:, 1]]
                self.add(key1, key2, mkpts1, mkpts2)
        return self


class MatchingStorage(Storage):
    def add(self, path1: FilePath, path2: FilePath, idxs: np.ndarray) -> None:
        raise NotImplementedError

    def __iter__(self) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
        raise NotImplementedError

    def get(self, path1: FilePath, path2: FilePath) -> np.ndarray:
        raise NotImplementedError

    def to_memory(self) -> InMemoryMatchingStorage:
        raise NotImplementedError

    def import_from(self, storage: MatchingStorage) -> None:
        raise NotImplementedError


class TwoViewGeometryStorage(Storage):
    def add(
        self, path1: FilePath, path2: FilePath, idxs: np.ndarray, F: np.ndarray
    ) -> None:
        raise NotImplementedError

    def get(self, path1: FilePath, path2: FilePath) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        idxs : np.ndarray
        F : np.ndarray
        """
        raise NotImplementedError

    def to_memory(self) -> InMemoryTwoViewGeometryStorage:
        raise NotImplementedError


class Line2DFeatureStorage(Storage):
    def add(self, path: FilePath, segs: np.ndarray, descinfos: Any) -> None:
        raise NotImplementedError

    def get(self, path: FilePath) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def to_memory(self) -> InMemoryLine2DFeatureStorage:
        raise NotImplementedError

    def get_endpoints(self, path: FilePath) -> np.ndarray:
        raise NotImplementedError

    def get_endpoints_with_interpolation(
        self, path: FilePath, n: int = 10
    ) -> np.ndarray:
        raise NotImplementedError


class Line2DSegmentStorage(Storage):
    def add(self, path: FilePath, segs: np.ndarray) -> None:
        raise NotImplementedError

    def get(self, path: FilePath) -> np.ndarray:
        raise NotImplementedError

    def to_memory(self) -> InMemoryLine2DSegmentStorage:
        raise NotImplementedError

    def get_endpoints(self, path: FilePath) -> np.ndarray:
        raise NotImplementedError

    def get_endpoints_with_interpolation(
        self, path: FilePath, n: int = 10
    ) -> np.ndarray:
        raise NotImplementedError

    def from_line2d_feature_storage(
        self, storage: Line2DFeatureStorage
    ) -> Line2DSegmentStorage:
        raise NotImplementedError


class InMemoryLine2DFeatureStorage(Line2DFeatureStorage):
    def __init__(self):
        self.segments: dict[str, np.ndarray] = {}
        self.descinfos: dict[str, Any] = {}

    def add(self, path: str | Path, segs: np.ndarray, descinfos: Any) -> None:
        name = Path(path).name
        self.segments[name] = segs.copy()
        if isinstance(descinfos, (list, tuple)):
            self.descinfos[name] = [deepcopy(d) for d in descinfos]
        elif isinstance(descinfos, dict):
            self.descinfos[name] = {k: deepcopy(v) for k, v in descinfos.items()}
        else:
            self.descinfos[name] = deepcopy(descinfos)

    def get(self, path: str | Path) -> tuple[np.ndarray, Any]:
        name = Path(path).name
        if isinstance(self.descinfos[name], (list, tuple)):
            return (
                self.segments[name].copy(),
                [deepcopy(d) for d in self.descinfos[name]],
            )
        elif isinstance(self.descinfos[name], dict):
            return (
                self.segments[name].copy(),
                {k: deepcopy(v) for k, v in self.descinfos[name].items()},
            )
        else:
            return self.segments[name].copy(), deepcopy(self.descinfos[name])

    def to_memory(self) -> InMemoryLine2DFeatureStorage:
        return self

    def get_endpoints(self, path: FilePath) -> np.ndarray:
        name = Path(path).name
        segs = self.segments.get(name)
        if segs is None:
            return np.empty((0, 2))
        points_a = segs[:, 0:2].copy()
        points_b = segs[:, 2:4].copy()
        return np.concatenate([points_a, points_b])

    def get_endpoints_with_interpolation(
        self, path: FilePath, n: int = 10
    ) -> np.ndarray:
        name = Path(path).name
        segs = self.segments.get(name)
        if segs is None:
            return np.empty((0, 2))

        xs = np.linspace(segs[:, 0], segs[:, 2], num=n)
        ys = np.linspace(segs[:, 1], segs[:, 3], num=n)
        return np.hstack([xs.T.flatten()[None].T, ys.T.flatten()[None].T])


class InMemoryLine2DSegmentStorage(Line2DSegmentStorage):
    def __init__(self):
        self.segments: dict[str, np.ndarray] = {}

    def add(self, path: str | Path, segs: np.ndarray) -> None:
        name = Path(path).name
        self.segments[name] = segs.copy()

    def get(self, path: str | Path) -> np.ndarray:
        name = Path(path).name
        return self.segments[name].copy()

    def to_memory(self) -> InMemoryLine2DSegmentStorage:
        return self

    def get_endpoints(self, path: FilePath) -> np.ndarray:
        name = Path(path).name
        segs = self.segments.get(name)
        if segs is None:
            return np.empty((0, 2))
        points_a = segs[:, 0:2].copy()
        points_b = segs[:, 2:4].copy()
        return np.concatenate([points_a, points_b])

    def get_endpoints_with_interpolation(
        self, path: FilePath, n: int = 10
    ) -> np.ndarray:
        name = Path(path).name
        segs = self.segments.get(name)
        if segs is None:
            return np.empty((0, 2))

        xs = np.linspace(segs[:, 0], segs[:, 2], num=n)
        ys = np.linspace(segs[:, 1], segs[:, 3], num=n)
        return np.hstack([xs.T.flatten()[None].T, ys.T.flatten()[None].T])

    def from_line2d_feature_storage(
        self, storage: Line2DFeatureStorage
    ) -> InMemoryLine2DSegmentStorage:
        storage = storage.to_memory()
        for name, seg in storage.segments.items():
            assert name not in self.segments
            self.segments[name] = seg.copy()
        return self


class InMemoryLocalFeatureStorage(LocalFeatureStorage):
    def __init__(self):
        self.lafs: dict[str, np.ndarray] = {}
        self.keypoints: dict[str, np.ndarray] = {}
        self.scores: dict[str, np.ndarray] = {}
        self.descriptors: dict[str, np.ndarray] = {}

    def check_size(self) -> None:
        b = 0
        b += sum([v.nbytes for _, v in self.lafs.items()])
        b += sum([v.nbytes for _, v in self.keypoints.items()])
        b += sum([v.nbytes for _, v in self.scores.items()])
        b += sum([v.nbytes for _, v in self.descriptors.items()])
        print(f"{self.__class__.__name__}: size={b / 1024 / 1024}")
        dtypes = []
        for k in self.lafs.keys():
            dtypes = [
                self.lafs[k].dtype,
                self.keypoints[k].dtype,
                self.scores[k].dtype,
                self.descriptors[k].dtype,
            ]
            break
        print(f"{self.__class__.__name__}: dtypes={dtypes}")

    def add(
        self,
        imagefile: FilePath,
        outputs: LocalFeatureExtractionOutputs,
    ) -> None:
        lafs, kpts, scores, descs = outputs
        imagefile = Path(imagefile)
        key = imagefile.name
        self.lafs[key] = lafs.copy()
        self.keypoints[key] = kpts.copy()
        self.scores[key] = scores.copy()
        self.descriptors[key] = descs.copy()
        if SHOW_MEM_USAGE:
            self.check_size()

    def get(self, path: FilePath) -> LocalFeatureExtractionOutputs:
        key = Path(path).name
        lafs = self.lafs[key].copy()
        kpts = self.keypoints[key].copy()
        scores = self.scores[key].copy()
        descs = self.descriptors[key].copy()
        return lafs, kpts, scores, descs

    def to_memory(self) -> InMemoryLocalFeatureStorage:
        return self

    def to_keypoint_storage(self) -> InMemoryKeypointStorage:
        storage = InMemoryKeypointStorage()
        storage.keypoints = {k: v.copy() for k, v in self.keypoints.items()}
        return storage


class HDF5LocalFeatureStorage(LocalFeatureStorage):
    def __init__(self, feature_dir: DirPath):
        self.feature_dir = Path(feature_dir)
        self.feature_dir.mkdir(parents=True, exist_ok=True)

    @property
    def lafs_file(self) -> Path:
        return self.feature_dir / "lafs.h5"

    @property
    def kpts_file(self) -> Path:
        return self.feature_dir / "keypoints.h5"

    @property
    def scores_file(self) -> Path:
        return self.feature_dir / "scores.h5"

    @property
    def descs_file(self) -> Path:
        return self.feature_dir / "descriptors.h5"

    def add(
        self,
        imagefile: FilePath,
        outputs: LocalFeatureExtractionOutputs,
    ) -> None:
        self._write_hdf5(imagefile, outputs)

    def get(self, path: FilePath) -> LocalFeatureExtractionOutputs:
        key = Path(path).name

        with (
            h5py.File(self.lafs_file, mode="r") as f_lafs,
            h5py.File(self.kpts_file, mode="r") as f_kpts,
            h5py.File(self.scores_file, mode="r") as f_scores,
            h5py.File(self.descs_file, mode="r") as f_descs,
        ):
            lafs: np.ndarray = f_lafs[key][...]  # type: ignore
            kpts: np.ndarray = f_kpts[key][...]  # type: ignore
            scores: np.ndarray = f_scores[key][...]  # type: ignore
            descs: np.ndarray = f_descs[key][...]  # type: ignore
        return lafs, kpts, scores, descs

    def to_memory(self) -> InMemoryLocalFeatureStorage:
        lafs, kpts, scores, descs = {}, {}, {}, {}
        with h5py.File(self.lafs_file, mode="r") as f:
            for k, d in f.items():
                lafs[k] = d[...]
        with h5py.File(self.kpts_file, mode="r") as f:
            for k, d in f.items():
                kpts[k] = d[...]
        with h5py.File(self.scores_file, mode="r") as f:
            for k, d in f.items():
                scores[k] = d[...]
        with h5py.File(self.descs_file, mode="r") as f:
            for k, d in f.items():
                descs[k] = d[...]
        storage = InMemoryLocalFeatureStorage()
        storage.lafs = lafs
        storage.keypoints = kpts
        storage.scores = scores
        storage.descriptors = descs
        return storage

    def _write_hdf5(
        self,
        imagefile: FilePath,
        outputs: LocalFeatureExtractionOutputs,
    ) -> None:
        imagefile = Path(imagefile)

        key = imagefile.name

        with (
            h5py.File(self.lafs_file, mode="a") as f_lafs,
            h5py.File(self.kpts_file, mode="a") as f_kpts,
            h5py.File(self.scores_file, mode="a") as f_scores,
            h5py.File(self.descs_file, mode="a") as f_descs,
        ):
            lafs, kpts, scores, descs = outputs
            f_lafs[key] = lafs
            f_kpts[key] = kpts
            f_scores[key] = scores
            f_descs[key] = descs


class InMemoryKeypointStorage(KeypointStorage):
    def __init__(self):
        self.keypoints: dict[str, np.ndarray] = {}

    def add(self, path: FilePath, kpts: np.ndarray) -> None:
        key = Path(path).name
        self.keypoints[key] = kpts.copy()

    def get(self, path: FilePath) -> np.ndarray:
        key = Path(path).name
        return self.keypoints[key].copy()

    def to_memory(self) -> InMemoryKeypointStorage:
        return self

    def fill_empty_keypoints(self, keys: list[str]) -> None:
        for key in keys:
            key = Path(key).name
            if key in self.keypoints:
                continue
            self.add(key, np.empty((0, 2), dtype=np.float32))
            print(f"Add empty keypoints for {key}")


class InMemoryMatchedKeypointStorage(MatchedKeypointStorage):
    def __init__(self):
        self.matched_keypoints: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        self.scores: dict[str, dict[str, np.ndarray]] = {}

    def __iter__(
        self,
    ) -> Iterator[tuple[str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
        for key1, group in self.matched_keypoints.items():
            yield key1, group

    def add(
        self,
        path1: FilePath,
        path2: FilePath,
        kpts1: np.ndarray,
        kpts2: np.ndarray,
        scores: Optional[np.ndarray] = None,
    ) -> None:
        key1 = Path(path1).name
        key2 = Path(path2).name
        if key1 not in self.matched_keypoints:
            self.matched_keypoints[key1] = {}
        self.matched_keypoints[key1][key2] = (kpts1.copy(), kpts2.copy())
        if scores is not None:
            if key1 not in self.scores:
                self.scores[key1] = {}
            self.scores[key1][key2] = scores.copy()

    def get(self, path1: FilePath, path2: FilePath) -> tuple[np.ndarray, np.ndarray]:
        key1 = Path(path1).name
        key2 = Path(path2).name
        mkpts1, mkpts2 = self.matched_keypoints[key1][key2]
        return mkpts1.copy(), mkpts2.copy()

    def get_scores(self, path1: FilePath, path2: FilePath) -> Optional[np.ndarray]:
        key1 = Path(path1).name
        key2 = Path(path2).name
        if not self.has_scores(key1, key2):
            return None
        return self.scores[key1][key2].copy()

    def has(self, path1: FilePath, path2: FilePath) -> bool:
        key1 = Path(path1).name
        key2 = Path(path2).name
        if key1 not in self.matched_keypoints:
            return False
        if key2 not in self.matched_keypoints[key1]:
            return False
        return True

    def has_scores(self, path1: FilePath, path2: FilePath) -> bool:
        key1 = Path(path1).name
        key2 = Path(path2).name
        if key1 not in self.scores:
            return False
        if key2 not in self.scores[key1]:
            return False
        return True

    def to_memory(self) -> InMemoryMatchedKeypointStorage:
        return self

    def fill_empty_matches(self, pairs: list[tuple[str, str]]) -> None:
        for path1, path2 in pairs:
            if self.has(path1, path2):
                continue
            self.add(
                path1,
                path2,
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
            )
            print(f"Add empty matches for ({path1}, {path2})")

    def clone_subset(
        self, image_paths: list[FilePath]
    ) -> InMemoryMatchedKeypointStorage:
        storage = InMemoryMatchedKeypointStorage()
        uses = set([Path(path).name for path in image_paths])
        matched_keypoints = {
            key1: {
                key2: (mkpts1.copy(), mkpts2.copy())
                for key2, (mkpts1, mkpts2) in vs.items()
                if key2 in uses
            }
            for key1, vs in self.matched_keypoints.items()
            if key1 in uses
        }
        scores = {
            key1: {key2: scores.copy() for key2, scores in vs.items() if key2 in uses}
            for key1, vs in self.scores.items()
            if key1 in uses
        }
        storage.matched_keypoints = matched_keypoints
        storage.scores = scores
        return storage


class InMemoryMatchingStorage(MatchingStorage):
    def __init__(self):
        self.matches: dict[str, dict[str, np.ndarray]] = {}

    def __iter__(self) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
        for key1, group in self.matches.items():
            yield key1, group

    def has(self, path1: FilePath, path2: FilePath) -> bool:
        key1 = Path(path1).name
        key2 = Path(path2).name
        if key1 not in self.matches:
            return False
        if key2 not in self.matches[key1]:
            return False
        return True

    def add(self, path1: FilePath, path2: FilePath, idxs: np.ndarray) -> None:
        key1 = Path(path1).name
        key2 = Path(path2).name
        if key1 not in self.matches:
            self.matches[key1] = {}
        self.matches[key1][key2] = idxs.copy()

    def get(self, path1: FilePath, path2: FilePath) -> np.ndarray:
        key1 = Path(path1).name
        key2 = Path(path2).name
        return self.matches[key1][key2].copy()

    def get_unique(self, path1: FilePath, path2: FilePath, scene: Scene) -> np.ndarray:
        key1 = Path(path1).name
        key2 = Path(path2).name
        i = scene.short_key_to_idx(key1)
        j = scene.short_key_to_idx(key2)
        if i < j:
            return self.get(path1, path2)
        idxs = self.get(path2, path1)
        return idxs[:, ::-1].copy()

    def to_memory(self) -> InMemoryMatchingStorage:
        return self

    def import_from(self, storage: MatchingStorage) -> None:
        self.matches = {}
        for key1, group in storage:
            for key2, idxs in group.items():
                self.add(key1, key2, idxs)


class HDF5MatchingStorage(MatchingStorage):
    def __init__(self, feature_dir: DirPath):
        self.feature_dir = Path(feature_dir)

    @property
    def match_file(self) -> Path:
        return self.feature_dir / "matches.h5"

    def __iter__(self) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
        with h5py.File(self.match_file, "r") as f_match:
            for key1, group in f_match.items():
                yield key1, {key2: d2[...] for key2, d2 in group.items()}

    def add(self, path1: FilePath, path2: FilePath, idxs: np.ndarray) -> None:
        self._write_hdf5(path1, path2, idxs)

    def get(self, path1: FilePath, path2: FilePath) -> np.ndarray:
        path1 = Path(path1)
        path2 = Path(path2)
        key1 = path1.name
        key2 = path2.name

        with h5py.File(self.match_file, mode="r") as f_match:
            idxs: np.ndarray = f_match[key1][key2][...]  # type: ignore
        return idxs

    def to_memory(self) -> InMemoryMatchingStorage:
        matches = {}
        with h5py.File(self.match_file, mode="r") as f:
            for key1, d1 in f.items():
                matches[key1] = {}
                for key2, d2 in d1.items():
                    matches[key1][key2] = d2[...]
        storage = InMemoryMatchingStorage()
        storage.matches = matches
        return storage

    def import_from(self, storage: MatchingStorage) -> None:
        if self.match_file.exists():
            self.match_file.unlink()
        for key1, group in storage:
            for key2, idxs in group.items():
                self.add(key1, key2, idxs)

    def _write_hdf5(
        self, path1: FilePath, path2: FilePath, idxs: np.ndarray, mode: str = "a"
    ) -> None:
        path1 = Path(path1)
        path2 = Path(path2)
        key1 = path1.name
        key2 = path2.name

        with h5py.File(self.match_file, mode=mode) as f_match:
            group = f_match.require_group(key1)
            group.create_dataset(key2, data=idxs)


class InMemoryTwoViewGeometryStorage(TwoViewGeometryStorage):
    def __init__(self):
        self.inliers: dict[str, dict[str, np.ndarray]] = {}
        self.Fs: dict[str, dict[str, np.ndarray]] = {}
        self.count = 0

    def add(
        self, path1: FilePath, path2: FilePath, idxs: np.ndarray, F: np.ndarray
    ) -> None:
        key1 = Path(path1).name
        key2 = Path(path2).name
        if key1 not in self.inliers:
            self.inliers[key1] = {}
            self.Fs[key1] = {}
        self.inliers[key1][key2] = idxs.copy()
        self.Fs[key1][key2] = F.copy()
        self.count += 1

    def get(self, path1: FilePath, path2: FilePath) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        idxs : np.ndarray
        F : np.ndarray
        """
        key1 = Path(path1).name
        key2 = Path(path2).name
        return (self.inliers[key1][key2].copy(), self.Fs[key1][key2].copy())

    def remove(self, path1: FilePath, path2: FilePath) -> None:
        key1 = Path(path1).name
        key2 = Path(path2).name
        del self.inliers[key1][key2]
        if not self.inliers[key1]:
            del self.inliers[key1]
        print(f"[InMemoryTwoViewGeometryStorage] Removed: ({key1}, {key2})")

    def to_memory(self) -> InMemoryTwoViewGeometryStorage:
        return self


def concat_keypoints(
    k_storage_list: list[KeypointStorage],
) -> InMemoryKeypointStorage:
    k_storage = InMemoryKeypointStorage()
    for _k_storage in k_storage_list:
        _k_storage = _k_storage.to_memory()
        for key in _k_storage.keypoints.keys():
            kpts = _k_storage.get(key)
            if key in k_storage.keypoints:
                _kpts = k_storage.get(key)
                kpts = np.vstack([_kpts, kpts])
                k_storage.add(key, kpts)
            else:
                k_storage.add(key, kpts)
    return k_storage


def concat_matched_keypoints(
    mk_storage_list: list[MatchedKeypointStorage], use_score_if_exists: bool = True
) -> InMemoryMatchedKeypointStorage:
    mk_storage = InMemoryMatchedKeypointStorage()
    for _mk_storage in mk_storage_list:
        for key1, group in _mk_storage:
            for key2, (kpts1, kpts2) in group.items():
                if use_score_if_exists:
                    scores = _mk_storage.get_scores(key1, key2)
                else:
                    scores = None

                if mk_storage.has(key1, key2):
                    _kpts1, _kpts2 = mk_storage.get(key1, key2)
                    _scores = mk_storage.get_scores(key1, key2)
                    kpts1 = np.vstack([_kpts1, kpts1])
                    kpts2 = np.vstack([_kpts2, kpts2])
                    if scores is not None:
                        assert _scores is not None
                        scores = np.concatenate([_scores, scores])
                    mk_storage.add(key1, key2, kpts1, kpts2, scores=scores)
                else:
                    mk_storage.add(key1, key2, kpts1, kpts2, scores=scores)
    return mk_storage


def convert_matched_keypoints_to_keypoints(
    matched_keypoint_storage: MatchedKeypointStorage,
    keypoint_storage: Optional[KeypointStorage] = None,
    matching_storage: Optional[MatchingStorage] = None,
    apply_round: bool = True,
) -> tuple[KeypointStorage, MatchingStorage]:
    """Convert matched keypoints to keypoints and matchings that are merged over unique matches"""
    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts = defaultdict(int)
    for key1, group in matched_keypoint_storage:
        for key2 in group.keys():
            mkpts1, mkpts2 = group[key2]
            n_matches = len(mkpts1)
            assert len(mkpts1) == len(mkpts2)
            kpts[key1].append(mkpts1)
            kpts[key2].append(mkpts2)
            current_match = torch.arange(n_matches).reshape(-1, 1).repeat(1, 2)
            current_match[:, 0] += total_kpts[key1]
            current_match[:, 1] += total_kpts[key2]
            total_kpts[key1] += n_matches
            total_kpts[key2] += n_matches
            match_indexes[key1][key2] = current_match

    for k in kpts.keys():
        if apply_round:
            kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
        else:
            kpts[k] = np.concatenate(kpts[k], axis=0)

    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(
            torch.from_numpy(kpts[k]), dim=0, return_inverse=True
        )
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()

    for k1, group in match_indexes.items():
        for k2, m in group.items():
            m2 = deepcopy(m)
            m2[:, 0] = unique_match_idxs[k1][m2[:, 0]]
            m2[:, 1] = unique_match_idxs[k2][m2[:, 1]]
            _kpts1 = unique_kpts[k1][m2[:, 0]]
            _kpts2 = unique_kpts[k2][m2[:, 1]]
            if _kpts1.ndim == 1:
                _kpts1 = _kpts1[None]
            if _kpts2.ndim == 1:
                _kpts2 = _kpts2[None]
            mkpts = np.concatenate([_kpts1, _kpts2], axis=1)

            if len(mkpts) == 0:
                out_match[k1][k2] = np.empty((0, 2))
                continue

            unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts), dim=0)
            m2_semiclean = m2[unique_idxs_current]
            unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
            m2_semiclean = m2_semiclean[unique_idxs_current1]
            unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
            m2_semiclean2 = m2_semiclean[unique_idxs_current2]
            out_match[k1][k2] = m2_semiclean2.numpy()

    keypoint_storage = keypoint_storage or InMemoryKeypointStorage()
    for key, kpts in unique_kpts.items():
        # TODO: "key" should be a path
        keypoint_storage.add(key, kpts)

    matching_storage = matching_storage or InMemoryMatchingStorage()
    for key1, group in out_match.items():
        for key2, idx in group.items():
            # TODO: "key" should be a path
            matching_storage.add(key1, key2, idx)

    # show_keypoint_matching_coverage(keypoint_storage, matching_storage)
    return keypoint_storage, matching_storage


def show_keypoint_matching_coverage(
    keypoint_storage: KeypointStorage, matching_storage: MatchingStorage
):
    for key1, group in matching_storage:
        kpts1 = keypoint_storage.get(key1)
        print("=================")
        print(f"[{key1}] keypoints: {len(kpts1)}")
        kpts1_matching_counts = {i: 0 for i in range(len(kpts1))}
        for key2, idxs in group.items():
            for i in idxs[:, 0]:
                kpts1_matching_counts[i] += 1
        num_keypoints_having_matches = sum(
            [1 if v > 0 else 0 for v in kpts1_matching_counts.values()]
        )
        num_matches = sum([v for v in kpts1_matching_counts.values()])
        max_matches = max([v for v in kpts1_matching_counts.values()])
        min_matches = min([v for v in kpts1_matching_counts.values()])
        avg_matches = np.array([v for v in kpts1_matching_counts.values()]).mean()
        avg_matches_effective = num_matches / num_keypoints_having_matches
        top10_matches = list(
            sorted(
                [(k, v) for k, v in kpts1_matching_counts.items()],
                key=lambda x: x[1],
                reverse=True,
            )
        )[:10]
        print(f"  - Keypoints:                          {len(kpts1)}")
        print(f"  - Keypoints having matches:           {num_keypoints_having_matches}")
        print(f"  - Matches:                            {num_matches}")
        print(f"  - Max matches per keypoint:           {max_matches}")
        print(f"  - Min matches per keypoint:           {min_matches}")
        print(f"  - Avg matches per keypoint:           {avg_matches}")
        print(f"  - Avg matches per effective keypoint: {avg_matches_effective}")
        print(
            f"  - Top10 match counts:                 {[v for _, v in top10_matches]}"
        )
        print(kpts1)


def fuse_matching_sets_early(
    paired_storages: list[tuple[LocalFeatureStorage, MatchingStorage]], scene: Scene
) -> tuple[InMemoryLocalFeatureStorage, InMemoryMatchingStorage]:
    concat_f_storage = InMemoryLocalFeatureStorage()
    concat_m_storage = InMemoryMatchingStorage()

    f_storages = []
    m_storages = []
    for f_storage, m_storage in paired_storages:
        f_storages.append(f_storage.to_memory())
        m_storages.append(m_storage.to_memory())

    if len(f_storages) == len(m_storages) == 1:
        return f_storages[0], m_storages[0]

    assert len(f_storages) > 1 and len(m_storages) > 1

    kpt_offsets = defaultdict(list)
    for path in scene.image_paths:
        k1 = Path(path).name
        lafs_list = []
        kpts_list = []
        scores_list = []
        descs_list = []
        num_total_kpts = 0
        for f_storage in f_storages:
            if k1 in f_storage.keypoints:
                lafs_list.append(f_storage.lafs[k1])
                kpts_list.append(f_storage.keypoints[k1])
                scores_list.append(f_storage.scores[k1])
                descs_list.append(f_storage.descriptors[k1])
                kpt_offsets[k1].append(num_total_kpts)
                num_total_kpts += len(f_storage.keypoints[k1])
            else:
                kpt_offsets[k1].append(None)

        n_cat = len(kpts_list)
        assert n_cat == len(lafs_list) == len(scores_list) == len(descs_list)

        concat_f_storage.lafs[k1] = np.concatenate(lafs_list)
        concat_f_storage.keypoints[k1] = np.concatenate(kpts_list)
        concat_f_storage.scores[k1] = np.concatenate(scores_list)
        concat_f_storage.descriptors[k1] = np.concatenate(descs_list)

    for path in scene.image_paths:
        k1 = Path(path).name
        for i, m_storage in enumerate(m_storages):
            if k1 not in m_storage.matches:
                continue
            for k2 in m_storage.matches[k1].keys():
                idxs = m_storage.matches[k1][k2].copy()
                assert kpt_offsets[k1][i] is not None
                assert kpt_offsets[k2][i] is not None
                idxs[:, 0] += int(kpt_offsets[k1][i])
                idxs[:, 1] += int(kpt_offsets[k2][i])
                if k1 not in concat_m_storage.matches:
                    concat_m_storage.add(k1, k2, idxs)
                    continue
                if k2 not in concat_m_storage.matches[k1]:
                    concat_m_storage.add(k1, k2, idxs)
                    continue
                concat_m_storage.matches[k1][k2] = np.concatenate(
                    [concat_m_storage.matches[k1][k2], idxs]
                )

    return concat_f_storage, concat_m_storage


def fuse_matching_sets_late(
    paired_storages: list[tuple[KeypointStorage, MatchingStorage]], scene: Scene
) -> tuple[InMemoryKeypointStorage, InMemoryMatchingStorage]:
    concat_f_storage = InMemoryKeypointStorage()
    concat_m_storage = InMemoryMatchingStorage()

    f_storages = []
    m_storages = []
    for f_storage, m_storage in paired_storages:
        f_storages.append(f_storage.to_memory())
        m_storages.append(m_storage.to_memory())

    if len(f_storages) == len(m_storages) == 1:
        return f_storages[0], m_storages[0]

    assert len(f_storages) > 1 and len(m_storages) > 1

    kpt_offsets = defaultdict(list)
    for path in scene.image_paths:
        k1 = Path(path).name
        kpts_list = []
        num_total_kpts = 0
        for f_storage in f_storages:
            if k1 in f_storage.keypoints:
                kpts_list.append(f_storage.keypoints[k1])
                kpt_offsets[k1].append(num_total_kpts)
                num_total_kpts += len(f_storage.keypoints[k1])
            else:
                kpt_offsets[k1].append(None)

        n_cat = len(kpts_list)
        concat_f_storage.keypoints[k1] = np.concatenate(kpts_list)

    for path in scene.image_paths:
        k1 = Path(path).name
        for i, m_storage in enumerate(m_storages):
            if k1 not in m_storage.matches:
                continue
            for k2 in m_storage.matches[k1].keys():
                idxs = m_storage.matches[k1][k2].copy()
                assert kpt_offsets[k1][i] is not None
                assert kpt_offsets[k2][i] is not None
                idxs[:, 0] += int(kpt_offsets[k1][i])
                idxs[:, 1] += int(kpt_offsets[k2][i])
                if k1 not in concat_m_storage.matches:
                    concat_m_storage.add(k1, k2, idxs)
                    continue
                if k2 not in concat_m_storage.matches[k1]:
                    concat_m_storage.add(k1, k2, idxs)
                    continue
                concat_m_storage.matches[k1][k2] = np.concatenate(
                    [concat_m_storage.matches[k1][k2], idxs]
                )

    return concat_f_storage, concat_m_storage


def fuse_line2d_matching_sets_late(
    paired_storages: list[tuple[Line2DSegmentStorage, MatchingStorage]], scene: Scene
) -> tuple[InMemoryLine2DSegmentStorage, InMemoryMatchingStorage]:
    concat_s_storage = InMemoryLine2DSegmentStorage()
    concat_m_storage = InMemoryMatchingStorage()

    s_storages = []
    m_storages = []
    for s_storage, m_storage in paired_storages:
        s_storages.append(s_storage.to_memory())
        m_storages.append(m_storage.to_memory())

    if len(s_storages) == len(m_storages) == 1:
        return s_storages[0], m_storages[0]

    assert len(s_storages) > 1 and len(m_storages) > 1
    raise NotImplementedError("TODO: Support multiple line2d extractors and matchers")


def get_unique_idxs(A: torch.Tensor, dim: int = 0) -> torch.Tensor:
    # https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
    unique, idx, counts = torch.unique(
        A, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    return first_indices


def keep_mutual_matches_only(
    dists: torch.Tensor, idxs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    first_indices = get_unique_idxs(idxs[:, 1])
    dists = dists[first_indices]
    idxs = idxs[first_indices]
    return dists, idxs


def filter_local_features_by_segmentation_mask(
    storage: LocalFeatureStorage, scene: Scene
) -> LocalFeatureStorage:
    assert isinstance(storage, InMemoryLocalFeatureStorage)
    if scene.segmentation_mask_images is None:
        return storage
    if len(scene.segmentation_mask_images) == 0:
        return storage

    for key in storage.lafs.keys():
        path = scene.short_key_to_image_path(key)

        mask = scene.get_segmentation_mask(path)
        if mask is None:
            continue
        h, w = mask.shape

        lafs, kpts, scores, descs = storage.get(key)
        keeps = []
        for kp in kpts:
            x = max(0, int(kp[0]))
            y = max(0, int(kp[1]))
            x = min(x, w - 1)
            y = min(y, h - 1)

            if mask[y, x] == 0:
                keeps.append(False)
            else:
                keeps.append(True)

        keeps = np.array(keeps, dtype=bool)
        lafs = lafs[keeps]
        kpts = kpts[keeps]
        scores = scores[keeps]
        descs = descs[keeps]

        print(f"Filter by segmentation ({key}): {len(keeps)} -> {len(kpts)}")
        storage.add(key, (lafs, kpts, scores, descs))
    return storage


def filter_local_features_by_mask_regions(
    storage: LocalFeatureStorage, scene: Scene
) -> LocalFeatureStorage:
    assert isinstance(storage, InMemoryLocalFeatureStorage)
    for key in storage.lafs.keys():
        path = scene.short_key_to_image_path(key)

        mask_bboxes = scene.get_mask_regions(path)
        if len(mask_bboxes) == 0:
            continue

        lafs, kpts, scores, descs = storage.get(key)
        for bbox in mask_bboxes:
            left, up, right, bottom = tuple(bbox)
            removed = (
                (left <= kpts[:, 0])
                & (kpts[:, 0] <= right)
                & (up <= kpts[:, 1])
                & (kpts[:, 1] <= bottom)
            )
            keep = ~removed
            lafs = lafs[keep]
            kpts = kpts[keep]
            scores = scores[keep]
            descs = descs[keep]

        storage.add(key, (lafs, kpts, scores, descs))
    return storage


def filter_matched_keypoints_by_mask_regions(
    storage: InMemoryMatchedKeypointStorage, scene: Scene
) -> InMemoryMatchedKeypointStorage:
    for key1, group in storage:
        for key2, (kpts1, kpts2) in group.items():
            scores = storage.get_scores(key1, key2)

            path1 = scene.short_key_to_image_path(key1)
            path2 = scene.short_key_to_image_path(key2)

            mask_bboxes = scene.get_mask_regions(path1)
            if len(mask_bboxes) == 0:
                continue

            for bbox in mask_bboxes:
                left, up, right, bottom = tuple(bbox)
                removed = (
                    (left <= kpts1[:, 0])
                    & (kpts1[:, 0] <= right)
                    & (up <= kpts1[:, 1])
                    & (kpts1[:, 1] <= bottom)
                )
                keep = ~removed
                kpts1 = kpts1[keep]
                kpts2 = kpts2[keep]
                if scores is not None:
                    scores = scores[keep]

            storage.add(key1, key2, kpts1, kpts2, scores=scores)
    return storage
