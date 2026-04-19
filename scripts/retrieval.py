from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: image retrieval from extracted features")
    parser.add_argument("--features-dir", default="data/features", help="Input features directory")
    parser.add_argument("--retrieval-dir", default="data/retrieval", help="Output retrieval directory")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k neighbors per image")
    return parser.parse_args()


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms
    return normalized @ normalized.T


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    features_dir = Path(args.features_dir)
    retrieval_dir = Path(args.retrieval_dir)
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    features_index_path = features_dir / "features_index.json"
    if not features_index_path.exists():
        raise FileNotFoundError(f"Missing features index: {features_index_path}")

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

    top_k = max(int(args.top_k), 1)
    retrieval_rows: list[dict] = []
    for i, image_id in enumerate(image_ids):
        row_sim = sim[i].copy()
        row_sim[i] = -1.0
        neighbor_idx = np.argsort(-row_sim)[:top_k]
        neighbors = [
            {"image_id": image_ids[j], "score": float(row_sim[j])}
            for j in neighbor_idx
            if row_sim[j] >= 0.0
        ]
        retrieval_rows.append({"query_image_id": image_id, "neighbors": neighbors})

    retrieval_index = {
        "num_queries": len(retrieval_rows),
        "top_k": top_k,
        "retrieval": retrieval_rows,
    }

    retrieval_index_path = retrieval_dir / "retrieval_index.json"
    with open(retrieval_index_path, "w") as f:
        json.dump(retrieval_index, f, indent=2)

    avg_neighbors = 0.0
    if retrieval_rows:
        avg_neighbors = sum(len(r["neighbors"]) for r in retrieval_rows) / len(retrieval_rows)

    duration = time.perf_counter() - t0

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("scene_reconstruction_dvc")

    with mlflow.start_run(run_name="retrieval"):
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("features_dir", str(features_dir))

        mlflow.log_metric("num_images", len(image_ids))
        mlflow.log_metric("num_queries", len(retrieval_rows))
        mlflow.log_metric("avg_neighbors", round(avg_neighbors, 4))
        mlflow.log_metric("execution_time", round(duration, 4))

        mlflow.log_artifacts(str(retrieval_dir), artifact_path="retrieval")

    print(
        json.dumps(
            {
                "num_queries": len(retrieval_rows),
                "avg_neighbors": round(avg_neighbors, 4),
                "retrieval_index": str(retrieval_index_path),
                "execution_time": round(duration, 4),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
