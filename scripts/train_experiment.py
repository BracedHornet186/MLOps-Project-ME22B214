"""
tests/train_experiment.py
────────────────────────────────────────────────────────────────────────────
Usage:
    python3 tests/train_experiment.py \
        --config conf/mast3r.yaml \
        --datasets ETs stairs \
        --experiment-name scene_reconstruction
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# ── Project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.config import PipelineConfig, SubmissionConfig
from scripts.data import DEFAULT_DATASET_DIR, IMC2025TrainData
import utils.imc25.metric as imc25_metric


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def registration_rate(submission_df: pd.DataFrame) -> float:
    def is_valid(r_str: str) -> bool:
        try:
            vals = [float(x) for x in str(r_str).split(";")]
            return len(vals) == 9 and not any(v != v for v in vals)
        except Exception:
            return False

    valid = submission_df["rotation_matrix"].apply(is_valid).sum()
    return float(valid / max(len(submission_df), 1))


def run_pipeline_and_collect_metrics(
    conf: SubmissionConfig,
    datasets: Optional[list[str]],
    submission_csv: Path,
):
    from scripts.kernel import run_and_save_submission
    from scripts.distributed import DistConfig

    if datasets:
        conf.datasets_to_use = datasets

    t_start = time.perf_counter()

    run_and_save_submission(
        conf,
        env_name="local",
        data_root_dir=DEFAULT_DATASET_DIR,
        dist_conf=DistConfig.single(),
    )

    elapsed = time.perf_counter() - t_start

    assert submission_csv.exists(), "submission.csv not found"

    # ── Evaluate ─────────────────────────────────────────
    user_df = pd.read_csv(submission_csv)
    gt_df = IMC2025TrainData.create(DEFAULT_DATASET_DIR).df

    final_scores_tuple, dataset_scores_tuple = imc25_metric.score(
        gt_csv=DEFAULT_DATASET_DIR / "train_labels.csv",
        user_csv=submission_csv,
        thresholds_csv=DEFAULT_DATASET_DIR / "train_thresholds.csv",
        mask_csv=None,
        inl_cf=0,
        strict_cf=-1,
        verbose=True,
    )

    final_score = final_scores_tuple[0]
    dataset_scores = dataset_scores_tuple[0]

    per_dataset = {ds: float(sc) for ds, sc in dataset_scores.items()}

    return float(final_score), per_dataset, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Main runner (NO MLFLOW)
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    config_path: str,
    datasets: Optional[list[str]],
):
    config_path = Path(config_path)
    submission_csv = ROOT / "submission.csv"

    pipeline_conf = PipelineConfig.load_config(config_path)
    submission_conf = SubmissionConfig(
        pipeline=pipeline_conf,
        target_data_type="imc2025train",
    )

    log.info("Running inference...")

    overall_maa, per_dataset, elapsed = run_pipeline_and_collect_metrics(
        submission_conf,
        datasets,
        submission_csv,
    )

    submission_df = pd.read_csv(submission_csv)

    reg_rate = registration_rate(submission_df)
    n_scenes = (
        submission_df.groupby(["dataset", "scene"]).ngroups
        if "scene" in submission_df.columns
        else 0
    )

    # ── Print results (test-friendly)
    print("\n" + "=" * 50)
    print("INFERENCE RESULTS")
    print("=" * 50)
    print(f"mAA (overall): {overall_maa:.4f}")
    print(f"Registration rate: {reg_rate:.4f}")
    print(f"Latency: {elapsed:.2f}s")
    print(f"Total images: {len(submission_df)}")
    print(f"Scenes: {n_scenes}")
    print("\nPer-dataset scores:")
    for ds, score in per_dataset.items():
        print(f"  {ds}: {score:.4f}")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Inference-only pipeline (no MLflow)")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--datasets", nargs="*", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    run_inference(
        config_path=args.config,
        datasets=args.datasets,
    )


if __name__ == "__main__":
    main()