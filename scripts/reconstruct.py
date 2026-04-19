from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6: placeholder COLMAP-style reconstruction from matches")
    parser.add_argument("--matches-dir", default="data/matches", help="Input matches directory")
    parser.add_argument("--reconstruction-dir", default="data/reconstruction", help="Output reconstruction directory")
    parser.add_argument("--min-inlier-matches", type=int, default=8, help="Minimum matches to keep an edge")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    matches_dir = Path(args.matches_dir)
    reconstruction_dir = Path(args.reconstruction_dir)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    matches_index_path = matches_dir / "matches_index.json"
    if not matches_index_path.exists():
        raise FileNotFoundError(f"Missing matches index: {matches_index_path}")

    with open(matches_index_path) as f:
        matches_index = json.load(f)

    min_inliers = int(args.min_inlier_matches)
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

    with mlflow.start_run(run_name="reconstruct"):
        mlflow.log_param("min_inlier_matches", min_inliers)

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
