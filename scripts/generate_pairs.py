from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: generate shortlist pairs from retrieval results")
    parser.add_argument("--retrieval-dir", default="data/retrieval", help="Input retrieval directory")
    parser.add_argument("--pairs-dir", default="data/pairs", help="Output pairs directory")
    parser.add_argument("--max-pairs-per-query", type=int, default=5, help="Maximum pairs emitted per query")
    return parser.parse_args()


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    retrieval_dir = Path(args.retrieval_dir)
    pairs_dir = Path(args.pairs_dir)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    retrieval_index_path = retrieval_dir / "retrieval_index.json"
    if not retrieval_index_path.exists():
        raise FileNotFoundError(f"Missing retrieval index: {retrieval_index_path}")

    with open(retrieval_index_path) as f:
        retrieval_index = json.load(f)

    retrieval_rows = retrieval_index.get("retrieval", [])
    max_pairs = max(int(args.max_pairs_per_query), 1)

    pairs_set: set[tuple[str, str]] = set()
    for row in retrieval_rows:
        query = row["query_image_id"]
        for neighbor in row.get("neighbors", [])[:max_pairs]:
            cand = neighbor["image_id"]
            if cand == query:
                continue
            pairs_set.add(canonical_pair(query, cand))

    pairs = [{"img1": a, "img2": b} for a, b in sorted(pairs_set)]

    pairs_index = {
        "num_pairs": len(pairs),
        "max_pairs_per_query": max_pairs,
        "pairs": pairs,
    }

    pairs_index_path = pairs_dir / "pairs_index.json"
    with open(pairs_index_path, "w") as f:
        json.dump(pairs_index, f, indent=2)

    duration = time.perf_counter() - t0

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("scene_reconstruction_dvc")

    with mlflow.start_run(run_name="generate_pairs"):
        mlflow.log_param("max_pairs_per_query", max_pairs)
        mlflow.log_param("retrieval_dir", str(retrieval_dir))

        mlflow.log_metric("num_queries", len(retrieval_rows))
        mlflow.log_metric("num_pairs", len(pairs))
        mlflow.log_metric("execution_time", round(duration, 4))

        mlflow.log_artifacts(str(pairs_dir), artifact_path="pairs")

    print(
        json.dumps(
            {
                "num_pairs": len(pairs),
                "pairs_index": str(pairs_index_path),
                "execution_time": round(duration, 4),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
