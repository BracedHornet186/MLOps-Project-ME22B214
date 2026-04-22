from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mlflow

from config import load_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: per-scene pair generation from retrieval results")
    parser.add_argument("--config", default="conf/mast3r.yaml")
    parser.add_argument("--retrieval-dir", default="data/retrieval")
    parser.add_argument("--pairs-dir", default="data/pairs")
    parser.add_argument("--max-pairs-per-query", type=int, default=None)
    return parser.parse_args()


def resolve_max_pairs_per_query(imc2025_conf) -> int:
    asmk_pair_limits: list[int] = []
    global_desc_topks: list[int] = []

    def _visit(shortlist_conf) -> None:
        if (
            shortlist_conf.type == "mast3r_retrieval_asmk"
            and shortlist_conf.mast3r_retrieval_asmk_make_pairs_fps_k
        ):
            asmk_pair_limits.append(int(shortlist_conf.mast3r_retrieval_asmk_make_pairs_fps_k))
        if shortlist_conf.type == "global_desc" and shortlist_conf.global_desc_topk:
            global_desc_topks.append(int(shortlist_conf.global_desc_topk))
        if shortlist_conf.type == "ensemble" and shortlist_conf.ensemble:
            for child in shortlist_conf.ensemble.shortlist_generators:
                _visit(child)

    _visit(imc2025_conf.shortlist_generator)

    if asmk_pair_limits:
        return max(1, max(asmk_pair_limits))
    if global_desc_topks:
        return max(1, max(global_desc_topks))
    return 5


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _discover_scenes(retrieval_dir: Path) -> list[tuple[str, str]]:
    scenes: list[tuple[str, str]] = []
    if not retrieval_dir.exists():
        return scenes
    for ds_dir in sorted(retrieval_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        for scene_dir in sorted(ds_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            if (scene_dir / "retrieval_index.json").exists():
                scenes.append((ds_dir.name, scene_dir.name))
    return scenes


def _generate_pairs_scene(
    retrieval_dir: Path,
    pairs_dir: Path,
    dataset: str,
    scene: str,
    max_pairs: int,
) -> int:
    retrieval_index_path = retrieval_dir / dataset / scene / "retrieval_index.json"
    with open(retrieval_index_path) as f:
        retrieval_index = json.load(f)

    retrieval_rows = retrieval_index.get("retrieval", [])
    pairs_set: set[tuple[str, str]] = set()
    for row in retrieval_rows:
        query = row["query_image_id"]
        for neighbor in row.get("neighbors", [])[:max_pairs]:
            cand = neighbor["image_id"]
            if cand != query:
                pairs_set.add(canonical_pair(query, cand))

    pairs = [{"img1": a, "img2": b} for a, b in sorted(pairs_set)]
    pairs_index = {
        "dataset": dataset,
        "scene": scene,
        "num_pairs": len(pairs),
        "max_pairs_per_query": max_pairs,
        "pairs": pairs,
    }

    scene_pairs_dir = pairs_dir / dataset / scene
    scene_pairs_dir.mkdir(parents=True, exist_ok=True)
    with open(scene_pairs_dir / "pairs_index.json", "w") as f:
        json.dump(pairs_index, f, indent=2)

    return len(pairs)


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    conf = load_pipeline_config(args.config)
    if conf.pipeline.type != "imc2025" or conf.pipeline.imc2025_pipeline is None:
        raise ValueError(f"Expected an imc2025 pipeline config, got type={conf.pipeline.type}")
    imc2025_conf = conf.pipeline.imc2025_pipeline

    retrieval_dir = Path(args.retrieval_dir)
    pairs_dir = Path(args.pairs_dir)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    resolved_max_pairs = resolve_max_pairs_per_query(imc2025_conf)
    max_pairs = max(int(args.max_pairs_per_query), 1) if args.max_pairs_per_query is not None else resolved_max_pairs

    scenes = _discover_scenes(retrieval_dir)
    if not scenes:
        raise FileNotFoundError(f"No scene retrieval indexes found under {retrieval_dir}")

    total_pairs = 0
    scene_pair_counts: list[dict] = []

    for dataset, scene in scenes:
        n = _generate_pairs_scene(retrieval_dir, pairs_dir, dataset, scene, max_pairs)
        total_pairs += n
        scene_pair_counts.append({"dataset": dataset, "scene": scene, "num_pairs": n})

    duration = time.perf_counter() - t0

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("scene_reconstruction_dvc")
    parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")
    run_tags = {"mlflow.parentRunId": parent_run_id} if parent_run_id else None

    with mlflow.start_run(run_name="generate_pairs", nested=True, tags=run_tags):
        mlflow.log_param("max_pairs_per_query", max_pairs)
        mlflow.log_param("retrieval_dir", str(retrieval_dir))
        mlflow.log_param("config_path", args.config)
        mlflow.log_param("num_scenes", len(scenes))

        mlflow.log_metric("num_scenes", len(scenes))
        mlflow.log_metric("num_pairs", total_pairs)
        mlflow.log_metric("execution_time", round(duration, 4))

        for s in scene_pair_counts:
            mlflow.log_metric(f"num_pairs_{s['dataset']}_{s['scene']}", s["num_pairs"])

        mlflow.log_artifacts(str(pairs_dir), artifact_path="pairs")

    print(json.dumps({
        "num_scenes": len(scenes),
        "num_pairs": total_pairs,
        "pairs_dir": str(pairs_dir),
        "execution_time": round(duration, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
