from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import mlflow

from config import load_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5: per-scene deterministic placeholder matching")
    parser.add_argument("--config", default="conf/mast3r.yaml")
    parser.add_argument("--pairs-dir", default="data/pairs")
    parser.add_argument("--keypoints-dir", default="data/keypoints")
    parser.add_argument("--matches-dir", default="data/matches")
    parser.add_argument("--min-match-threshold", type=int, default=None)
    return parser.parse_args()


def resolve_min_match_threshold(imc2025_conf) -> int:
    candidates: list[int] = []
    for matcher in imc2025_conf.point_tracking_matchers:
        if matcher.type == "mast3r_hybrid" and matcher.mast3r_hybrid:
            if matcher.mast3r_hybrid.dense_min_matches is not None:
                candidates.append(int(matcher.mast3r_hybrid.dense_min_matches))
            if matcher.mast3r_hybrid.sparse_min_matches is not None:
                candidates.append(int(matcher.mast3r_hybrid.sparse_min_matches))
        if matcher.type == "mast3r_sparse" and matcher.mast3r_sparse:
            if matcher.mast3r_sparse.min_matches is not None:
                candidates.append(int(matcher.mast3r_sparse.min_matches))
        if matcher.type == "vggt" and matcher.vggt:
            if matcher.vggt.min_matches is not None:
                candidates.append(int(matcher.vggt.min_matches))
    if candidates:
        return max(1, min(candidates))
    return 8


def pair_hash(a: str, b: str) -> int:
    h = hashlib.sha256(f"{a}|{b}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


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


def _match_scene(
    pairs_dir: Path,
    keypoints_dir: Path,
    matches_dir: Path,
    dataset: str,
    scene: str,
    min_match_threshold: int,
) -> dict:
    pairs_index_path = pairs_dir / dataset / scene / "pairs_index.json"
    keypoints_index_path = keypoints_dir / dataset / scene / "keypoints_index.json"

    if not pairs_index_path.exists():
        raise FileNotFoundError(f"Missing pairs index: {pairs_index_path}")
    if not keypoints_index_path.exists():
        raise FileNotFoundError(f"Missing keypoints index: {keypoints_index_path}")

    with open(pairs_index_path) as f:
        pairs_index = json.load(f)
    with open(keypoints_index_path) as f:
        keypoints_index = json.load(f)

    keypoint_counts: dict[str, int] = {
        rec["image_id"]: int(rec.get("num_keypoints", 0))
        for rec in keypoints_index.get("records", [])
    }

    matches: list[dict] = []
    total_matches = 0
    pairs_with_matches = 0

    for pair in pairs_index.get("pairs", []):
        img1 = pair["img1"]
        img2 = pair["img2"]
        n1 = keypoint_counts.get(img1, 0)
        n2 = keypoint_counts.get(img2, 0)
        cap = max(min(n1, n2), 0)

        if cap == 0:
            num_matches = 0
        else:
            num_matches = min(cap, 8 + (pair_hash(img1, img2) % max(cap, 1)))

        if num_matches < min_match_threshold:
            num_matches = 0

        if num_matches > 0:
            pairs_with_matches += 1
        total_matches += num_matches

        matches.append({
            "img1": img1,
            "img2": img2,
            "num_matches": int(num_matches),
            "matched_idx": list(range(int(num_matches))),
        })

    matches_index = {
        "dataset": dataset,
        "scene": scene,
        "num_pairs": len(matches),
        "pairs_with_matches": pairs_with_matches,
        "total_matches": total_matches,
        "min_match_threshold": min_match_threshold,
        "matches": matches,
    }

    scene_matches_dir = matches_dir / dataset / scene
    scene_matches_dir.mkdir(parents=True, exist_ok=True)
    with open(scene_matches_dir / "matches_index.json", "w") as f:
        json.dump(matches_index, f, indent=2)

    return {"num_pairs": len(matches), "pairs_with_matches": pairs_with_matches, "total_matches": total_matches}


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    conf = load_pipeline_config(args.config)
    if conf.pipeline.type != "imc2025" or conf.pipeline.imc2025_pipeline is None:
        raise ValueError(f"Expected an imc2025 pipeline config, got type={conf.pipeline.type}")
    imc2025_conf = conf.pipeline.imc2025_pipeline

    min_match_threshold = (
        max(int(args.min_match_threshold), 1)
        if args.min_match_threshold is not None
        else resolve_min_match_threshold(imc2025_conf)
    )

    pairs_dir = Path(args.pairs_dir)
    keypoints_dir = Path(args.keypoints_dir)
    matches_dir = Path(args.matches_dir)
    matches_dir.mkdir(parents=True, exist_ok=True)

    scenes = _discover_scenes(pairs_dir)
    if not scenes:
        raise FileNotFoundError(f"No scene pair indexes found under {pairs_dir}")

    total_pairs = 0
    total_matches_sum = 0
    total_pairs_with_matches = 0
    scene_stats: list[dict] = []

    for dataset, scene in scenes:
        stats = _match_scene(pairs_dir, keypoints_dir, matches_dir, dataset, scene, min_match_threshold)
        total_pairs += stats["num_pairs"]
        total_matches_sum += stats["total_matches"]
        total_pairs_with_matches += stats["pairs_with_matches"]
        scene_stats.append({"dataset": dataset, "scene": scene, **stats})

    avg_matches = total_matches_sum / total_pairs if total_pairs else 0.0
    duration = time.perf_counter() - t0

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("scene_reconstruction_dvc")
    parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")
    run_tags = {"mlflow.parentRunId": parent_run_id} if parent_run_id else None

    with mlflow.start_run(run_name="match", nested=True, tags=run_tags):
        mlflow.log_param("min_match_threshold", min_match_threshold)
        mlflow.log_param("config_path", args.config)
        mlflow.log_param("num_scenes", len(scenes))

        mlflow.log_metric("num_scenes", len(scenes))
        mlflow.log_metric("num_pairs", total_pairs)
        mlflow.log_metric("pairs_with_matches", total_pairs_with_matches)
        mlflow.log_metric("num_matches", total_matches_sum)
        mlflow.log_metric("avg_matches", round(avg_matches, 4))
        mlflow.log_metric("execution_time", round(duration, 4))

        for s in scene_stats:
            key = f"{s['dataset']}_{s['scene']}"
            mlflow.log_metric(f"num_matches_{key}", s["total_matches"])
            mlflow.log_metric(f"pairs_with_matches_{key}", s["pairs_with_matches"])

        mlflow.log_artifacts(str(matches_dir), artifact_path="matches")

    print(json.dumps({
        "num_scenes": len(scenes),
        "num_pairs": total_pairs,
        "num_matches": total_matches_sum,
        "matches_dir": str(matches_dir),
        "execution_time": round(duration, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
