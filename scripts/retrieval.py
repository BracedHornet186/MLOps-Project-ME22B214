from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mlflow
import numpy as np

from config import load_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: per-scene image retrieval from extracted features")
    parser.add_argument("--config", default="conf/mast3r.yaml")
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--retrieval-dir", default="data/retrieval")
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def resolve_top_k(imc2025_conf) -> int:
    global_desc_topks: list[int] = []
    fallback_topks: list[int] = []

    def _visit(shortlist_conf) -> None:
        if shortlist_conf.type == "global_desc" and shortlist_conf.global_desc_topk:
            global_desc_topks.append(int(shortlist_conf.global_desc_topk))
        if (
            shortlist_conf.type == "mast3r_retrieval_asmk"
            and shortlist_conf.mast3r_retrieval_asmk_make_pairs_fps_k
        ):
            fallback_topks.append(int(shortlist_conf.mast3r_retrieval_asmk_make_pairs_fps_k))
        if shortlist_conf.type == "ensemble" and shortlist_conf.ensemble:
            for child in shortlist_conf.ensemble.shortlist_generators:
                _visit(child)

    _visit(imc2025_conf.shortlist_generator)

    if global_desc_topks:
        return max(1, max(global_desc_topks))
    if fallback_topks:
        return max(1, max(fallback_topks))
    return 10


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms
    return normalized @ normalized.T


def _discover_scenes(features_dir: Path) -> list[tuple[str, str]]:
    """Return sorted (dataset, scene) pairs found under features_dir."""
    scenes: list[tuple[str, str]] = []
    if not features_dir.exists():
        return scenes
    for ds_dir in sorted(features_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        for scene_dir in sorted(ds_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            if (scene_dir / "features_index.json").exists():
                scenes.append((ds_dir.name, scene_dir.name))
    return scenes


def _retrieve_scene(
    features_dir: Path,
    retrieval_dir: Path,
    dataset: str,
    scene: str,
    top_k: int,
) -> dict:
    scene_features_dir = features_dir / dataset / scene
    features_index_path = scene_features_dir / "features_index.json"

    with open(features_index_path) as f:
        features_index = json.load(f)

    records = features_index.get("records", [])
    image_ids: list[str] = []
    vectors: list[np.ndarray] = []

    for rec in records:
        image_id = rec["image_id"]
        vector_path = Path(rec["vector_path"])
        if not vector_path.exists():
            continue
        image_ids.append(image_id)
        vectors.append(np.load(vector_path))

    if vectors:
        matrix = np.stack(vectors, axis=0)
        sim = cosine_similarity_matrix(matrix)
    else:
        sim = np.zeros((0, 0), dtype=np.float32)

    effective_k = min(top_k, max(len(image_ids) - 1, 0))
    retrieval_rows: list[dict] = []
    for i, image_id in enumerate(image_ids):
        row_sim = sim[i].copy()
        row_sim[i] = -1.0
        neighbor_idx = np.argsort(-row_sim)[:effective_k]
        neighbors = [
            {"image_id": image_ids[j], "score": float(row_sim[j])}
            for j in neighbor_idx
            if row_sim[j] >= 0.0
        ]
        retrieval_rows.append({"query_image_id": image_id, "neighbors": neighbors})

    retrieval_index = {
        "dataset": dataset,
        "scene": scene,
        "num_queries": len(retrieval_rows),
        "top_k": effective_k,
        "retrieval": retrieval_rows,
    }

    scene_retrieval_dir = retrieval_dir / dataset / scene
    scene_retrieval_dir.mkdir(parents=True, exist_ok=True)
    with open(scene_retrieval_dir / "retrieval_index.json", "w") as f:
        json.dump(retrieval_index, f, indent=2)

    avg_neighbors = (
        sum(len(r["neighbors"]) for r in retrieval_rows) / len(retrieval_rows)
        if retrieval_rows else 0.0
    )
    return {"num_images": len(image_ids), "num_queries": len(retrieval_rows), "avg_neighbors": avg_neighbors}


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    conf = load_pipeline_config(args.config)
    if conf.pipeline.type != "imc2025" or conf.pipeline.imc2025_pipeline is None:
        raise ValueError(f"Expected an imc2025 pipeline config, got type={conf.pipeline.type}")
    imc2025_conf = conf.pipeline.imc2025_pipeline

    features_dir = Path(args.features_dir)
    retrieval_dir = Path(args.retrieval_dir)
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    resolved_top_k = resolve_top_k(imc2025_conf)
    top_k = max(int(args.top_k), 1) if args.top_k is not None else resolved_top_k

    scenes = _discover_scenes(features_dir)
    if not scenes:
        raise FileNotFoundError(f"No scene feature indexes found under {features_dir}")

    total_images = 0
    total_queries = 0
    scene_stats: list[dict] = []

    for dataset, scene in scenes:
        stats = _retrieve_scene(features_dir, retrieval_dir, dataset, scene, top_k)
        total_images += stats["num_images"]
        total_queries += stats["num_queries"]
        scene_stats.append({"dataset": dataset, "scene": scene, **stats})

    avg_neighbors_global = (
        sum(s["avg_neighbors"] for s in scene_stats) / len(scene_stats)
        if scene_stats else 0.0
    )
    duration = time.perf_counter() - t0

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("scene_reconstruction_dvc")
    parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")
    run_tags = {"mlflow.parentRunId": parent_run_id} if parent_run_id else None

    with mlflow.start_run(run_name="retrieval", nested=True, tags=run_tags):
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("features_dir", str(features_dir))
        mlflow.log_param("config_path", args.config)
        mlflow.log_param("num_scenes", len(scenes))

        mlflow.log_metric("num_images", total_images)
        mlflow.log_metric("num_queries", total_queries)
        mlflow.log_metric("num_scenes", len(scenes))
        mlflow.log_metric("avg_neighbors", round(avg_neighbors_global, 4))
        mlflow.log_metric("execution_time", round(duration, 4))

        for s in scene_stats:
            key = f"{s['dataset']}_{s['scene']}"
            mlflow.log_metric(f"num_images_{key}", s["num_images"])
            mlflow.log_metric(f"avg_neighbors_{key}", round(s["avg_neighbors"], 4))

        mlflow.log_artifacts(str(retrieval_dir), artifact_path="retrieval")

    print(json.dumps({
        "num_scenes": len(scenes),
        "num_queries": total_queries,
        "avg_neighbors": round(avg_neighbors_global, 4),
        "retrieval_dir": str(retrieval_dir),
        "execution_time": round(duration, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
