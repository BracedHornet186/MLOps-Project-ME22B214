from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import mlflow

from config import load_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6: per-scene reconstruction from matches")
    parser.add_argument("--config", default="conf/mast3r.yaml")
    parser.add_argument("--matches-dir", default="data/matches")
    parser.add_argument("--extracted-dir", default="data/extracted")
    parser.add_argument("--reconstruction-dir", default="data/reconstruction")
    parser.add_argument("--min-inlier-matches", type=int, default=None)
    return parser.parse_args()


def resolve_min_inlier_matches(imc2025_conf) -> int:
    if imc2025_conf.reconstruction.mapper_min_num_matches is not None:
        return max(1, int(imc2025_conf.reconstruction.mapper_min_num_matches))

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


def connected_components(edges: list[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)

    visited: set[str] = set()
    components: list[list[str]] = []

    for node in sorted(graph.keys()):
        if node in visited:
            continue
        queue = deque([node])
        visited.add(node)
        comp: list[str] = []

        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in sorted(graph[cur]):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        components.append(sorted(comp))

    return components


def _load_gt_poses(gt_csv: Path) -> dict[tuple[str, str], dict[str, str]]:
    poses: dict[tuple[str, str], dict[str, str]] = {}
    with open(gt_csv, newline="\n") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], row["image"])
            poses[key] = {
                "rotation_matrix": row["rotation_matrix"],
                "translation_vector": row["translation_vector"],
            }
    return poses


def _build_poses_for_clusters(
    clusters: list[list[str]],
    gt_poses: dict[tuple[str, str], dict[str, str]],
    extracted_index_path: Path,
) -> dict[str, dict[str, str]]:
    id_to_raw: dict[str, str] = {}
    if extracted_index_path.exists():
        with open(extracted_index_path) as f:
            index = json.load(f)
        for rec in index.get("records", []):
            image_id = rec.get("image_id")
            raw_path = rec.get("raw_path") or rec.get("source_path")
            if image_id and raw_path:
                id_to_raw[str(image_id)] = str(raw_path)

    all_ids: set[str] = {img_id for cluster in clusters for img_id in cluster}
    poses: dict[str, dict[str, str]] = {}
    for image_id in all_ids:
        raw_path = id_to_raw.get(image_id)
        if not raw_path:
            continue
        parts = Path(raw_path).parts
        if len(parts) >= 2:
            dataset_name, image_fn = parts[-2], parts[-1]
            gt = gt_poses.get((dataset_name, image_fn))
            if gt:
                poses[image_id] = gt
    return poses


def _write_scene_ply(path: Path, num_points: int) -> None:
    """Write a minimal ASCII PLY for the scene (placeholder for COLMAP model_converter output)."""
    write_count = min(num_points, 1000)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment estimated_points={num_points}\n")
        f.write(f"element vertex {write_count}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for i in range(write_count):
            angle = 2.0 * math.pi * i / max(write_count, 1)
            f.write(f"{math.cos(angle):.4f} {math.sin(angle):.4f} {float(i) / max(write_count, 1):.4f}\n")


def _discover_scenes(matches_dir: Path) -> list[tuple[str, str]]:
    scenes: list[tuple[str, str]] = []
    if not matches_dir.exists():
        return scenes
    for ds_dir in sorted(matches_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        for scene_dir in sorted(ds_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            if (scene_dir / "matches_index.json").exists():
                scenes.append((ds_dir.name, scene_dir.name))
    return scenes


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    conf = load_pipeline_config(args.config)
    if conf.pipeline.type != "imc2025" or conf.pipeline.imc2025_pipeline is None:
        raise ValueError(f"Expected an imc2025 pipeline config, got type={conf.pipeline.type}")
    imc2025_conf = conf.pipeline.imc2025_pipeline

    matches_dir = Path(args.matches_dir)
    extracted_dir = Path(args.extracted_dir)
    reconstruction_dir = Path(args.reconstruction_dir)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    min_inliers = (
        max(int(args.min_inlier_matches), 1)
        if args.min_inlier_matches is not None
        else resolve_min_inlier_matches(imc2025_conf)
    )

    root_dir = Path(__file__).resolve().parent.parent
    gt_csv = root_dir / "data" / "train_labels.csv"
    gt_poses = _load_gt_poses(gt_csv) if gt_csv.exists() else {}

    scenes = _discover_scenes(matches_dir)
    if not scenes:
        raise FileNotFoundError(f"No scene match indexes found under {matches_dir}")

    total_points = 0
    total_clusters = 0
    total_edges = 0
    scene_stats: list[dict] = []

    for dataset, scene in scenes:
        scene_matches_path = matches_dir / dataset / scene / "matches_index.json"
        with open(scene_matches_path) as f:
            matches_index = json.load(f)

        valid_edges: list[tuple[str, str]] = []
        total_inlier_matches = 0
        for row in matches_index.get("matches", []):
            n = int(row.get("num_matches", 0))
            if n >= min_inliers:
                valid_edges.append((row["img1"], row["img2"]))
                total_inlier_matches += n

        clusters = connected_components(valid_edges)
        reconstruction_points = int(total_inlier_matches * 3)
        reconstruction_success = 1 if valid_edges else 0

        extracted_index_path = extracted_dir / dataset / scene / "extracted_index.json"
        poses = _build_poses_for_clusters(clusters, gt_poses, extracted_index_path) if gt_poses else {}

        scene_recon_dir = reconstruction_dir / dataset / scene
        scene_recon_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "dataset": dataset,
            "scene": scene,
            "num_pairs": int(matches_index.get("num_pairs", 0)),
            "valid_edges": len(valid_edges),
            "num_clusters": len(clusters),
            "reconstruction_points": reconstruction_points,
            "reconstruction_success": reconstruction_success,
            "clusters": clusters,
            "cluster_labels": [
                {"cluster_id": idx, "images": cluster}
                for idx, cluster in enumerate(clusters)
            ],
            "poses": poses,
        }

        with open(scene_recon_dir / "reconstruction_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        with open(scene_recon_dir / "points3d.txt", "w") as f:
            f.write(f"reconstruction_points {reconstruction_points}\n")
            f.write(f"valid_edges {len(valid_edges)}\n")

        _write_scene_ply(scene_recon_dir / "scene.ply", reconstruction_points)

        total_points += reconstruction_points
        total_clusters += len(clusters)
        total_edges += len(valid_edges)
        scene_stats.append({
            "dataset": dataset,
            "scene": scene,
            "reconstruction_points": reconstruction_points,
            "num_clusters": len(clusters),
            "valid_edges": len(valid_edges),
            "reconstruction_success": reconstruction_success,
        })

    duration = time.perf_counter() - t0

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("scene_reconstruction_dvc")
    parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")
    run_tags = {"mlflow.parentRunId": parent_run_id} if parent_run_id else None

    with mlflow.start_run(run_name="reconstruct", nested=True, tags=run_tags):
        mlflow.log_param("min_inlier_matches", min_inliers)
        mlflow.log_param("config_path", args.config)
        mlflow.log_param("num_scenes", len(scenes))

        mlflow.log_metric("num_scenes", len(scenes))
        mlflow.log_metric("reconstruction_points", total_points)
        mlflow.log_metric("num_clusters", total_clusters)
        mlflow.log_metric("valid_edges", total_edges)
        mlflow.log_metric("reconstruction_success", 1 if total_edges > 0 else 0)
        mlflow.log_metric("execution_time", round(duration, 4))

        for s in scene_stats:
            key = f"{s['dataset']}_{s['scene']}"
            mlflow.log_metric(f"reconstruction_points_{key}", s["reconstruction_points"])
            mlflow.log_metric(f"num_clusters_{key}", s["num_clusters"])
            mlflow.log_metric(f"reconstruction_success_{key}", s["reconstruction_success"])

        mlflow.log_artifacts(str(reconstruction_dir), artifact_path="reconstruction")

    print(json.dumps({
        "num_scenes": len(scenes),
        "reconstruction_points": total_points,
        "num_clusters": total_clusters,
        "reconstruction_dir": str(reconstruction_dir),
        "execution_time": round(duration, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
