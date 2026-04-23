from __future__ import annotations

import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import utils.imc25.metric
from scripts.config import PipelineConfig, SubmissionConfig
from scripts.data import DEFAULT_DATASET_DIR
from scripts.kernel import run_and_save_submission


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-c", "--conf", required=True)
    parser.add_argument("--env-name", default="local")
    parser.add_argument("--data-root-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument(
        "-p",
        "--preset-group",
        choices=("imc2025", "imc2024", "imc2024x", "pt"),
        default=None,
    )
    return parser.parse_args()


def evaluate(
    conf_path: str | Path,
    env_name: str,
    data_root_dir: str,
    datasets: list[str] | None = None,
    preset_group: str | None = None,
):
    conf = SubmissionConfig(
        pipeline=PipelineConfig.load_config(conf_path), target_data_type="imc2025train"
    )
    if preset_group == "imc2025":
        conf.datasets_to_use = [
            "amy_gardens",
            "ETs",
            "fbk_vineyard",
            "stairs",
        ]
    elif preset_group == "imc2024":
        conf.datasets_to_use = [
            "imc2023_haiper",
            "imc2023_heritage",
            "imc2023_theather_imc2024_church",
            "imc2024_dioscuri_baalshamin",
            "imc2024_lizard_pond",
        ]
    elif preset_group == "imc2024x":
        conf.datasets_to_use = [
            "imc2023_haiper",
            # "imc2023_heritage",
            "imc2023_theather_imc2024_church",
            "imc2024_dioscuri_baalshamin",
            "imc2024_lizard_pond",
        ]
    elif preset_group == "pt":
        conf.datasets_to_use = [
            "pt_brandenburg_british_buckingham",
            "pt_piazzasanmarco_grandplace",
            "pt_sacrecoeur_trevi_tajmahal",
            "pt_stpeters_stpauls",
        ]
    elif datasets:
        conf.datasets_to_use = datasets

    print(conf.datasets_to_use)

    start_time = time.time()
    run_and_save_submission(conf, env_name=env_name, data_root_dir=data_root_dir)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.02f} sec.")

    submission_csv = Path("submission.csv")
    assert submission_csv.exists()

    t = time.time()
    final_score, dataset_scores = utils.imc25.metric.score(
        gt_csv=DEFAULT_DATASET_DIR / "train_labels.csv",
        user_csv=submission_csv,
        thresholds_csv=DEFAULT_DATASET_DIR / "train_thresholds.csv",
        mask_csv=None,
        inl_cf=0,
        strict_cf=-1,
        verbose=True,
    )
    print(f"Computed metric in: {time.time() - t:.02f} sec.")
    print(f"final_score: {final_score}")
    print(f"dataset_scores: {dataset_scores}")


def main():
    args = parse_args()
    evaluate(
        args.conf,
        args.env_name,
        args.data_root_dir,
        args.datasets,
        preset_group=args.preset_group,
    )


if __name__ == "__main__":
    main()