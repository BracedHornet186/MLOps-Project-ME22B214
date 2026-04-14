from typing import Optional

import h5py
import numpy as np
import hloc.utils.io
from hloc.utils.parsers import names_to_pair

from data import FilePath
from storage import (
    InMemoryLocalFeatureStorage,
    InMemoryMatchedKeypointStorage,
    InMemoryMatchingStorage,
    InMemoryKeypointStorage,
    KeypointStorage,
    LocalFeatureStorage,
    MatchedKeypointStorage,
)


def export_local_features_to_hloc(
    storage: LocalFeatureStorage, hloc_feature_path: FilePath
) -> str:
    assert isinstance(storage, InMemoryLocalFeatureStorage)
    hloc_feature_path = str(hloc_feature_path)

    with h5py.File(hloc_feature_path, "a", libver="latest") as fd:
        for name in storage.lafs.keys():
            lafs, kpts, scores, descs = storage.get(name)
            pred = {
                "lafs": lafs,
                "keypoints": kpts,
                "scores": scores,
                "descriptors": descs,
            }
            if name in fd:
                del fd[name]
            grp = fd.create_group(name)
            for k, v in pred.items():
                grp.create_dataset(k, data=v)
            if "keypoints" in pred:
                # TODO
                # grp["keypoints"].attrs["uncertainty"] = uncertainty
                pass

    return hloc_feature_path


def import_local_features_from_hloc(
    hloc_feature_path: FilePath, storage: Optional[InMemoryLocalFeatureStorage] = None
) -> InMemoryLocalFeatureStorage:
    hloc_feature_path = str(hloc_feature_path)
    if storage is None:
        storage = InMemoryLocalFeatureStorage()

    with h5py.File(hloc_feature_path, "r", libver="latest") as fd:
        for name in fd.keys():
            lafs = fd[str(name)]["lafs"].__array__()  # type: ignore
            kpts = fd[str(name)]["keypoints"].__array__()  # type: ignore
            scores = fd[str(name)]["scores"].__array__()  # type: ignore
            descs = fd[str(name)]["descriptors"].__array__()  # type: ignore
            storage.add(name, (lafs, kpts, scores, descs))

    return storage


def export_keypoints_to_hloc(
    storage: KeypointStorage,
    hloc_feature_path: FilePath
) -> str:
    print(f"Exporting keypoints to {hloc_feature_path}")
    storage = storage.to_memory()
    hloc_feature_path = str(hloc_feature_path)

    with h5py.File(hloc_feature_path, "a", libver="latest") as fd:
        for name in storage.keypoints.keys():
            kpts = storage.get(name)
            pred = {
                "keypoints": kpts,
            }
            if name in fd:
                del fd[name]
            grp = fd.create_group(name)
            for k, v in pred.items():
                grp.create_dataset(k, data=v)
            if "keypoints" in pred:
                # TODO
                # grp["keypoints"].attrs["uncertainty"] = uncertainty
                pass
    return hloc_feature_path


def import_keypoints_from_hloc(
    hloc_feature_path: FilePath,
    storage: Optional[InMemoryKeypointStorage] = None,
    remove_empty_keypoints: bool = True,
) -> InMemoryKeypointStorage:
    hloc_feature_path = str(hloc_feature_path)
    if storage is None:
        storage = InMemoryKeypointStorage()

    with h5py.File(hloc_feature_path, "r", libver="latest") as fd:
        for name in fd.keys():
            kpts = fd[str(name)]["keypoints"].__array__()  # type: ignore
            if remove_empty_keypoints and len(kpts) == 0:
                print(
                    f"[import_keypoints_from_hloc] "
                    f"Skip importing keypoints of {name} because they are empty"
                )
                continue
            storage.add(name, kpts)
    return storage


def export_matched_keypoints_to_hloc(
    storage: MatchedKeypointStorage, hloc_match_path: FilePath
) -> str:
    print(f"Exporting matched keypoints to {hloc_match_path}")
    assert isinstance(storage, InMemoryMatchedKeypointStorage)
    hloc_match_path = str(hloc_match_path)

    with h5py.File(hloc_match_path, "a", libver="latest") as fd:
        for key1, group in storage:
            for key2, (mkpts1, mkpts2) in group.items():
                scores = storage.get_scores(key1, key2)
                assert scores is not None

                name = names_to_pair(key1, key2)
                if name in fd:
                    del fd[name]
                grp = fd.create_group(name)

                # Write dense matching output
                grp.create_dataset("keypoints0", data=mkpts1)
                grp.create_dataset("keypoints1", data=mkpts2)
                grp.create_dataset("scores", data=scores)

    return hloc_match_path


def import_matched_keypoints_from_hloc(
    hloc_match_path: FilePath, storage: Optional[InMemoryMatchedKeypointStorage] = None,
    remove_empty_keypoints: bool = True
) -> InMemoryMatchedKeypointStorage:
    print(f"Importing matched keypoints from {hloc_match_path}")
    hloc_match_path = str(hloc_match_path)
    if storage is None:
        storage = InMemoryMatchedKeypointStorage()

    with h5py.File(hloc_match_path, "a") as fd:
        for key1 in fd.keys():
            for key2 in fd[key1].keys():  # type: ignore
                name = names_to_pair(key1, key2)
                grp = fd[name]
                mkpts1 = grp["keypoints0"].__array__()  # type: ignore
                mkpts2 = grp["keypoints1"].__array__()  # type: ignore
                scores = grp["scores"].__array__()  # type: ignore
                if remove_empty_keypoints and len(mkpts1) == 0:
                    print(
                        f"[import_matched_keypoints_from_hloc] "
                        f"Skip importing keypoints of {name} because they are empty"
                    )
                    continue
                storage.add(key1, key2, mkpts1, mkpts2, scores=scores)
    return storage


def import_matching_from_hloc(
    hloc_match_path: FilePath, storage: Optional[InMemoryMatchingStorage] = None,
    remove_empty_matchings: bool = True
) -> InMemoryMatchingStorage:
    print(f"Importing matchings from {hloc_match_path}")
    hloc_match_path = str(hloc_match_path)
    if storage is None:
        storage = InMemoryMatchingStorage()

    with h5py.File(hloc_match_path, "r", libver="latest") as fd:
        for key1 in fd.keys():
            for key2 in fd[key1].keys():  # type: ignore
                name = names_to_pair(key1, key2)
                grp = fd[name]
                if "matches0" not in grp:
                    print(
                        f"[import_matching_from_hloc] "
                        f"Skip importing matchings of {name} because 'matches' is not in grp"
                    )
                    continue
                matches = grp["matches0"].__array__()  # type: ignore
                scores = grp["matching_scores0"].__array__()  # type: ignore
                if remove_empty_matchings and len(matches) == 0:
                    print(
                        f"[import_matching_from_hloc] "
                        f"Skip importing matchings of {name} because they are empty"
                    )
                    continue
                idx = np.where(matches != -1)[0]
                matches = np.stack([idx, matches[idx]], -1)
                scores = scores[idx]
                storage.add(key1, key2, matches)
    return storage


if __name__ == "__main__":
    from extractor import extract_all
    from pipeline import DetectorBasedPipeline, DetectorFreePipeline
    from pipelines.matching import run_detector_free_matching
    from utils.debug import get_first_scene, setup_data_and_pipeline

    if True:
        # Check local features
        data, pipeline = setup_data_and_pipeline(
            config_path="conf/pipeline/imc2024/lightglue_aliked/lightgluealiked-012.yaml",
            scene_names=["mttb"],
            max_num_samples=10,
        )
        assert isinstance(pipeline, DetectorBasedPipeline)
        scene = get_first_scene(data, pipeline)

        storage1 = InMemoryLocalFeatureStorage()
        extractor = pipeline.extractors[0]
        extract_all(extractor, scene, storage=storage1)

        path = export_local_features_to_hloc(storage1, "test_hloc.h5")

        storage2 = InMemoryLocalFeatureStorage()
        import_local_features_from_hloc("test_hloc.h5", storage=storage2)

        for key in storage1.lafs.keys():
            lafs1, kpts1, scores1, descs1 = storage1.get(key)
            lafs2, kpts2, scores2, descs2 = storage2.get(key)

            print(np.all(lafs1 == lafs2))
            print(np.all(kpts1 == kpts2))
            print(np.all(scores1 == scores2))
            print(np.all(descs1 == descs2))
            print("----")

    if True:
        # Check dense matching
        data, pipeline = setup_data_and_pipeline(
            config_path="conf/pipeline/imc2024/loftr/loftr-005.yaml",
            scene_names=["mttb"],
            max_num_samples=10,
        )
        assert isinstance(pipeline, DetectorFreePipeline)
        scene = get_first_scene(data, pipeline)
        pairs = pipeline.shortlist_generator(scene)
        print(f"# of pairs: {len(pairs)}")

        storage1 = InMemoryMatchedKeypointStorage()
        matcher = pipeline.matcher
        run_detector_free_matching(
            matcher, pairs, scene, matched_keypoint_storage=storage1
        )

        path = export_matched_keypoints_to_hloc(storage1, "test_match_hloc.h5")

        storage2 = InMemoryMatchedKeypointStorage()
        import_matched_keypoints_from_hloc("test_match_hloc.h5", storage2)

        for key1, group in storage1:
            for key2 in group.keys():
                mkpts1, mkpts2 = storage1.get(key1, key2)
                mkpts3, mkpts4 = storage2.get(key1, key2)
                print(np.all(mkpts1 == mkpts3))
                print(np.all(mkpts2 == mkpts4))
