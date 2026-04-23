from __future__ import annotations

from collections import defaultdict
from collections.abc import Generator
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from clusterings.base import Clustering
from data import arr_to_str, nan_R_str, nan_t_str
from scripts.data_schema import DataSchema
from pipelines.scene import Scene


def create_data_dict(
    data_schema: DataSchema,
    df: Optional[pd.DataFrame] = None,
    listfile_path: Optional[str] = None,
    ignore_gt_scene_label: bool = True,
) -> dict[str, dict[str, list[str | Path]]]:
    if df is None:
        assert listfile_path
        df = pd.read_csv(listfile_path)
    assert df is not None

    if ignore_gt_scene_label:
        df["scene"] = ["cluster0"] * len(df)

    data = {}
    for i in range(len(df)):
        row = df.iloc[i]
        image_path = data_schema.resolve_image_path(row)
        if not Path(image_path).exists():
            print(f"NotFound: {image_path}")
            continue

        dataset = row["dataset"]
        scene = row["scene"]

        if dataset not in data:
            data[dataset] = {}

        if scene not in data[dataset]:
            data[dataset][scene] = []

        data[dataset][scene].append(image_path)

    return data


def init_result_dict(
    data_dict: dict[str, dict[str, list[str | Path]]],
) -> tuple[dict[str, dict[str, dict]], int]:
    results = {}
    scene_count = 0
    for dataset in data_dict.keys():
        results[dataset] = {}
        for scene in data_dict[dataset]:
            results[dataset][scene] = {}
            scene_count += 1
    return results, scene_count


def init_result_dict_with_scene_clustering(
    data_dict: dict[str, dict[str, list[str | Path]]],
) -> dict[str, dict[str, dict]]:
    results = {}
    for dataset in data_dict.keys():
        results[dataset] = defaultdict(dict)
    return results


def iterate_scenes(
    data_dict: dict[str, dict[str, list[str | Path]]],
    data_schema: DataSchema,
    clustering: Clustering | None = None,
) -> Generator:
    for dataset, scenes in data_dict.items():
        if clustering:
            # Ignore pre-defined scene names in a csvfile
            image_paths = []
            for _image_paths in scenes.values():
                image_paths += _image_paths

            # New scene list will be created by clustering
            yield from clustering.run(image_paths).to_scene_list(dataset, data_schema)
        else:
            for scene, image_paths in scenes.items():
                image_dirs = [str(Path(path).parent) for path in image_paths]
                # All images from the same scene must be in the same directory
                assert len(set(image_dirs)) == 1
                image_dir = image_dirs[0]
                yield Scene(
                    dataset=dataset,
                    scene=scene,
                    image_paths=image_paths,  # type: ignore
                    image_dir=image_dir,
                    data_schema=data_schema,
                )


def results_to_submission_df(
    results: dict[str, dict[str, dict]],
    schema: Literal["imc2024", "imc2025"] = "imc2025",
) -> pd.DataFrame:
    if schema == "imc2024":
        data = {
            "image_path": [],
            "dataset": [],
            "scene": [],
            "rotation_matrix": [],
            "translation_vector": [],
        }

        for dataset, scene_results in results.items():
            for scene, scene_preds in scene_results.items():
                for image, pred in scene_preds.items():
                    R = pred["R"].reshape(-1)
                    t = pred["t"].reshape(-1)
                    data["image_path"].append(image)
                    data["dataset"].append(dataset)
                    data["scene"].append(scene)
                    data["rotation_matrix"].append(arr_to_str(R))
                    data["translation_vector"].append(arr_to_str(t))
    elif schema == "imc2025":
        print("results_to_submission_df (schema=imc2025)")
        print("-----------------------------------------")
        results = pre_clustered_scene_results_to_final_clustered_scene_results(results)

        data = {
            "image_id": [],
            "dataset": [],
            "scene": [],
            "image": [],
            "rotation_matrix": [],
            "translation_vector": [],
        }

        for dataset, scene_results in results.items():
            for scene, scene_preds in scene_results.items():
                for image, pred in scene_preds.items():
                    if np.isnan(pred["R"]).any() or np.isnan(pred["t"]).any():
                        R_str = nan_R_str()
                        t_str = nan_t_str()
                    else:
                        R = pred["R"].reshape(-1)
                        t = pred["t"].reshape(-1)
                        R_str = arr_to_str(R)
                        t_str = arr_to_str(t)

                    # scene_name = pred.get("cluster_name") or scene
                    data["image_id"].append(pred["metadata"].get("image_id"))
                    data["dataset"].append(dataset)
                    data["scene"].append(scene)
                    data["image"].append(image)
                    data["rotation_matrix"].append(R_str)
                    data["translation_vector"].append(t_str)

    else:
        raise NotImplementedError(schema)

    return pd.DataFrame(data)


def pre_clustered_scene_results_to_final_clustered_scene_results(
    results: dict[str, dict[str, dict]],
) -> dict[str, dict[str, dict]]:
    print("Pre-clustered results")
    for dataset, pre_scene_results in results.items():
        print(f"  * dataset={dataset}")
        for pre_scene, pre_scene_preds in pre_scene_results.items():
            print(f"    - scene={pre_scene} (#preds={len(pre_scene_preds)})")
    print("")
    final_results = {}
    for dataset, pre_scene_results in results.items():
        if dataset not in final_results:
            final_results[dataset] = {}
        for pre_scene, pre_scene_preds in pre_scene_results.items():
            for image, pred in pre_scene_preds.items():
                final_scene_name = pred.get("cluster_name") or pre_scene
                print(f" - {image}: {pre_scene} -> {final_scene_name}")
                if final_scene_name not in final_results[dataset]:
                    final_results[dataset][final_scene_name] = {}
                final_results[dataset][final_scene_name][image] = pred
    print("-----------------------------------------")
    print("Final-clustered results")
    for dataset, scene_results in final_results.items():
        print(f"  * dataset={dataset}")
        for scene, scene_preds in scene_results.items():
            print(f"    - scene={scene} (#preds={len(scene_preds)})")
    return final_results
