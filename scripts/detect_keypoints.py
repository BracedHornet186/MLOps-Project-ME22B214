from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import mlflow
import numpy as np

from config import load_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4: per-scene deterministic keypoint detection")
    parser.add_argument("--config", default="conf/mast3r.yaml")
    parser.add_argument("--extracted-dir", default="data/extracted")
    parser.add_argument("--pairs-dir", default="data/pairs")
    parser.add_argument("--keypoints-dir", default="data/keypoints")
    parser.add_argument("--min-keypoints", type=int, default=None)
    parser.add_argument("--max-keypoints", type=int, default=None)
    return parser.parse_args()


def resolve_min_keypoints(imc2025_conf) -> int:
    candidates: list[int] = []
    for matcher in imc2025_conf.point_tracking_matchers:
        hybrid = matcher.mast3r_hybrid
        if not hybrid:
            continue
        if hybrid.sparse_min_matches is not None:
            candidates.append(int(hybrid.sparse_min_matches))
        if hybrid.dense_min_matches is not None:
            candidates.append(int(hybrid.dense_min_matches))
    if candidates:
        return max(8, min(candidates))
    return 32


def resolve_max_keypoints(min_keypoints: int) -> int:
    return max(min_keypoints + 1, min_keypoints * 8)


def deterministic_keypoints(image_id: str, min_kpts: int, max_kpts: int) -> tuple[np.ndarray, np.ndarray]:
    digest = hashlib.sha256(image_id.encode("utf-8")).digest()
    span = max(max_kpts - min_kpts, 1)
    n = min_kpts + (digest[0] % span)

    points = []
    scores = []
    for i in range(n):
        d = digest[i % len(digest)]
        x = float((d + i * 17) % 1000) / 10.0
        y = float((d + i * 29) % 1000) / 10.0
        s = float((d % 100) / 100.0)
        points.append([x, y])
        scores.append(s)

    return np.asarray(points, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def _discover_scenes(pairs_dir: Path) -> list[tuple[str, str]]:
    scenes: list[tuple[str, str]] = []
    if not pairs_dir.exists():
        return scenes
    for ds_dir in sorted(pairs_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        for scene_dir in sorted(ds_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            if (scene_dir / "pairs_index.json").exists():
                scenes.append((ds_dir.name, scene_dir.name))
    return scenes


def _detect_scene(
    extracted_dir: Path,
    pairs_dir: Path,
    keypoints_dir: Path,
    dataset: str,
    scene: str,
    min_keypoints: int,
    max_keypoints: int,
) -> dict:
    extracted_index_path = extracted_dir / dataset / scene / "extracted_index.json"
    pairs_index_path = pairs_dir / dataset / scene / "pairs_index.json"

    if not extracted_index_path.exists():
        raise FileNotFoundError(f"Missing extracted index: {extracted_index_path}")
    if not pairs_index_path.exists():
        raise FileNotFoundError(f"Missing pairs index: {pairs_index_path}")

    with open(extracted_index_path) as f:
        extracted_index = json.load(f)
    with open(pairs_index_path) as f:
        pairs_index = json.load(f)

    valid_ids = {r["image_id"] for r in extracted_index.get("records", [])}
    needed_ids: set[str] = set()
    for pair in pairs_index.get("pairs", []):
        needed_ids.add(pair["img1"])
        needed_ids.add(pair["img2"])
    needed_ids = needed_ids & valid_ids

    scene_keypoints_dir = keypoints_dir / dataset / scene
    scene_keypoints_dir.mkdir(parents=True, exist_ok=True)

    keypoint_records: list[dict] = []
    total_keypoints = 0

    for image_id in sorted(needed_ids):
        points, scores = deterministic_keypoints(image_id, min_keypoints, max_keypoints)
        total_keypoints += int(points.shape[0])

        out_path = scene_keypoints_dir / f"{image_id}.npz"
        np.savez_compressed(out_path, keypoints=points, scores=scores)
        keypoint_records.append({
            "image_id": image_id,
            "keypoints_path": str(out_path),
            "num_keypoints": int(points.shape[0]),
        })

    keypoints_index = {
        "dataset": dataset,
        "scene": scene,
        "num_images": len(keypoint_records),
        "total_keypoints": total_keypoints,
        "records": keypoint_records,
    }
    with open(scene_keypoints_dir / "keypoints_index.json", "w") as f:
        json.dump(keypoints_index, f, indent=2)

    return {"num_images": len(keypoint_records), "total_keypoints": total_keypoints}


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    conf = load_pipeline_config(args.config)
    if conf.pipeline.type != "imc2025" or conf.pipeline.imc2025_pipeline is None:
        raise ValueError(f"Expected an imc2025 pipeline config, got type={conf.pipeline.type}")
    imc2025_conf = conf.pipeline.imc2025_pipeline

    min_keypoints = int(args.min_keypoints) if args.min_keypoints is not None else resolve_min_keypoints(imc2025_conf)
    max_keypoints = int(args.max_keypoints) if args.max_keypoints is not None else resolve_max_keypoints(min_keypoints)
    if max_keypoints <= min_keypoints:
        max_keypoints = min_keypoints + 1

    extracted_dir = Path(args.extracted_dir)
    pairs_dir = Path(args.pairs_dir)
    keypoints_dir = Path(args.keypoints_dir)
    keypoints_dir.mkdir(parents=True, exist_ok=True)

    scenes = _discover_scenes(pairs_dir)
    if not scenes:
        raise FileNotFoundError(f"No scene pair indexes found under {pairs_dir}")

    total_images = 0
    total_keypoints = 0
    scene_stats: list[dict] = []

    for dataset, scene in scenes:
        stats = _detect_scene(extracted_dir, pairs_dir, keypoints_dir, dataset, scene, min_keypoints, max_keypoints)
        total_images += stats["num_images"]
        total_keypoints += stats["total_keypoints"]
        scene_stats.append({"dataset": dataset, "scene": scene, **stats})

    avg_keypoints = total_keypoints / total_images if total_images else 0.0
    duration = time.perf_counter() - t0

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("scene_reconstruction_dvc")
    parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")
    run_tags = {"mlflow.parentRunId": parent_run_id} if parent_run_id else None

    with mlflow.start_run(run_name="detect_keypoints", nested=True, tags=run_tags):
        mlflow.log_param("min_keypoints", min_keypoints)
        mlflow.log_param("max_keypoints", max_keypoints)
        mlflow.log_param("config_path", args.config)
        mlflow.log_param("num_scenes", len(scenes))

        mlflow.log_metric("num_images", total_images)
        mlflow.log_metric("num_scenes", len(scenes))
        mlflow.log_metric("total_keypoints", total_keypoints)
        mlflow.log_metric("avg_keypoints", round(avg_keypoints, 4))
        mlflow.log_metric("execution_time", round(duration, 4))

        for s in scene_stats:
            key = f"{s['dataset']}_{s['scene']}"
            mlflow.log_metric(f"total_keypoints_{key}", s["total_keypoints"])

        mlflow.log_artifacts(str(keypoints_dir), artifact_path="keypoints")

    print(json.dumps({
        "num_scenes": len(scenes),
        "num_images": total_images,
        "total_keypoints": total_keypoints,
        "keypoints_dir": str(keypoints_dir),
        "execution_time": round(duration, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
