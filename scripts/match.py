from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mlflow
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5: deterministic placeholder matching for image pairs")
    parser.add_argument("--pairs-dir", default="data/pairs", help="Input pairs directory")
    parser.add_argument("--keypoints-dir", default="data/keypoints", help="Input keypoints directory")
    parser.add_argument("--matches-dir", default="data/matches", help="Output matches directory")
    parser.add_argument("--min-match-threshold", type=int, default=8, help="Minimum matches for reconstruction eligibility")
    return parser.parse_args()


def pair_hash(a: str, b: str) -> int:
    h = hashlib.sha256(f"{a}|{b}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    pairs_dir = Path(args.pairs_dir)
    keypoints_dir = Path(args.keypoints_dir)
    matches_dir = Path(args.matches_dir)
    matches_dir.mkdir(parents=True, exist_ok=True)

    pairs_index_path = pairs_dir / "pairs_index.json"
    keypoints_index_path = keypoints_dir / "keypoints_index.json"

    if not pairs_index_path.exists():
        raise FileNotFoundError(f"Missing pairs index: {pairs_index_path}")
    if not keypoints_index_path.exists():
        raise FileNotFoundError(f"Missing keypoints index: {keypoints_index_path}")

    with open(pairs_index_path) as f:
        pairs_index = json.load(f)
    with open(keypoints_index_path) as f:
        keypoints_index = json.load(f)

    keypoint_counts: dict[str, int] = {}
    for rec in keypoints_index.get("records", []):
        keypoint_counts[rec["image_id"]] = int(rec.get("num_keypoints", 0))

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

        if num_matches > 0:
            pairs_with_matches += 1
        total_matches += num_matches

        matches.append(
            {
                "img1": img1,
                "img2": img2,
                "num_matches": int(num_matches),
                "matched_idx": list(range(int(num_matches))),
            }
        )

    matches_index = {
        "num_pairs": len(matches),
        "pairs_with_matches": pairs_with_matches,
        "total_matches": total_matches,
        "min_match_threshold": args.min_match_threshold,
        "matches": matches,
    }

    matches_index_path = matches_dir / "matches_index.json"
    with open(matches_index_path, "w") as f:
        json.dump(matches_index, f, indent=2)

    avg_matches = 0.0
    if matches:
        avg_matches = total_matches / len(matches)

    duration = time.perf_counter() - t0

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("scene_reconstruction_dvc")

    with mlflow.start_run(run_name="match"):
        mlflow.log_param("min_match_threshold", args.min_match_threshold)

        mlflow.log_metric("num_pairs", len(matches))
        mlflow.log_metric("pairs_with_matches", pairs_with_matches)
        mlflow.log_metric("num_matches", total_matches)
        mlflow.log_metric("avg_matches", round(avg_matches, 4))
        mlflow.log_metric("execution_time", round(duration, 4))

        mlflow.log_artifacts(str(matches_dir), artifact_path="matches")

    print(
        json.dumps(
            {
                "num_pairs": len(matches),
                "num_matches": total_matches,
                "matches_index": str(matches_index_path),
                "execution_time": round(duration, 4),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
