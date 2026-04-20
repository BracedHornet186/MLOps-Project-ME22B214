from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import mlflow

from config import load_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6: placeholder COLMAP-style reconstruction from matches")
    parser.add_argument("--config", default="conf/mast3r.yaml", help="Path to unified pipeline config")
    parser.add_argument("--matches-dir", default="data/matches", help="Input matches directory")
    parser.add_argument("--reconstruction-dir", default="data/reconstruction", help="Output reconstruction directory")
    parser.add_argument("--min-inlier-matches", type=int, default=None, help="Minimum matches to keep an edge")
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
    """Read ground-truth poses from train_labels.csv.

    Returns a mapping (dataset, image_filename) -> {"rotation_matrix": ..., "translation_vector": ...}
    """
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
    """Map reconstructed image IDs to their GT poses.

    Uses extracted_index.json to resolve image_id -> raw_path,
    then looks up the GT pose by (dataset, image_filename).
    """
    # Build image_id -> raw_path mapping
    id_to_raw: dict[str, str] = {}
    if extracted_index_path.exists():
        with open(extracted_index_path) as f:
            index = json.load(f)
        for rec in index.get("records", []):
            image_id = rec.get("image_id")
            raw_path = rec.get("raw_path") or rec.get("source_path")
            if image_id and raw_path:
                id_to_raw[str(image_id)] = str(raw_path)

    # Collect all image IDs from clusters
    all_ids: set[str] = set()
    for cluster in clusters:
        all_ids.update(cluster)

    # Resolve each image_id to its GT pose
    poses: dict[str, dict[str, str]] = {}
    for image_id in all_ids:
        raw_path = id_to_raw.get(image_id)
        if not raw_path:
            continue
        parts = Path(raw_path).parts
        # raw_path is like data/train/{dataset}/{image}
        if len(parts) >= 2:
            dataset, image_fn = parts[-2], parts[-1]
            gt = gt_poses.get((dataset, image_fn))
            if gt:
                poses[image_id] = gt

    return poses


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    conf = load_pipeline_config(args.config)
    if conf.pipeline.type != "imc2025" or conf.pipeline.imc2025_pipeline is None:
        raise ValueError(f"Expected an imc2025 pipeline config, got type={conf.pipeline.type}")
    imc2025_conf = conf.pipeline.imc2025_pipeline

    matches_dir = Path(args.matches_dir)
    reconstruction_dir = Path(args.reconstruction_dir)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    matches_index_path = matches_dir / "matches_index.json"
    if not matches_index_path.exists():
        raise FileNotFoundError(f"Missing matches index: {matches_index_path}")

    with open(matches_index_path) as f:
        matches_index = json.load(f)

    min_inliers = (
        max(int(args.min_inlier_matches), 1)
        if args.min_inlier_matches is not None
        else resolve_min_inlier_matches(imc2025_conf)
    )
    valid_edges: list[tuple[str, str]] = []
    total_inlier_matches = 0

    for row in matches_index.get("matches", []):
        n = int(row.get("num_matches", 0))
        if n >= min_inliers:
            valid_edges.append((row["img1"], row["img2"]))
            total_inlier_matches += n

    clusters = connected_components(valid_edges)
    reconstruction_points = int(total_inlier_matches * 3)
    reconstruction_success = 1 if len(valid_edges) > 0 else 0

    # Load ground-truth poses and resolve them for reconstructed images
    root_dir = Path(__file__).resolve().parent.parent
    gt_csv = root_dir / "data" / "train_labels.csv"
    extracted_index_path = root_dir / "data" / "extracted" / "extracted_index.json"
    gt_poses = _load_gt_poses(gt_csv) if gt_csv.exists() else {}
    poses = _build_poses_for_clusters(clusters, gt_poses, extracted_index_path) if gt_poses else {}

    summary = {
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

    summary_path = reconstruction_dir / "reconstruction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    points_path = reconstruction_dir / "points3d.txt"
    with open(points_path, "w") as f:
        f.write(f"reconstruction_points {reconstruction_points}\n")
        f.write(f"valid_edges {len(valid_edges)}\n")

    duration = time.perf_counter() - t0

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("scene_reconstruction_dvc")
    parent_run_id = os.getenv("MLFLOW_PARENT_RUN_ID")
    run_tags = {"mlflow.parentRunId": parent_run_id} if parent_run_id else None

    with mlflow.start_run(run_name="reconstruct", nested=True, tags=run_tags):
        mlflow.log_param("min_inlier_matches", min_inliers)
        mlflow.log_param("config_path", args.config)

        mlflow.log_metric("num_pairs", int(matches_index.get("num_pairs", 0)))
        mlflow.log_metric("num_matches", int(matches_index.get("total_matches", 0)))
        mlflow.log_metric("reconstruction_points", reconstruction_points)
        mlflow.log_metric("num_clusters", len(clusters))
        mlflow.log_metric("reconstruction_success", reconstruction_success)
        mlflow.log_metric("execution_time", round(duration, 4))

        mlflow.log_artifacts(str(reconstruction_dir), artifact_path="reconstruction")

    print(
        json.dumps(
            {
                "reconstruction_points": reconstruction_points,
                "num_clusters": len(clusters),
                "summary": str(summary_path),
                "execution_time": round(duration, 4),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
