from __future__ import annotations

from argparse import ArgumentParser, Namespace

import numpy as np
import sklearn.metrics
import torch
import tqdm

from clusterings.config import ClusteringConfig
from clusterings.factory import create_clustering
from data import IMC2025TrainData


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-c", "--conf", required=True)
    parser.add_argument("-d", "--datasets", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda")

    conf = ClusteringConfig.from_file(args.conf)
    clustering = create_clustering(conf, device=device)

    scores = []
    data = IMC2025TrainData.create("data", datasets_to_use=args.datasets)
    dataset_groups = data.df.groupby("dataset").groups
    for dataset_name, idx in tqdm.tqdm(
        dataset_groups.items(), total=len(dataset_groups), desc="Clustering evaluation"
    ):
        dataset_df = data.df.loc[idx]
        uniq_scene_names = dataset_df["scene"].unique().tolist()
        uniq_scene_name_to_label = {
            scene_name: i for i, scene_name in enumerate(uniq_scene_names)
        }
        t = np.array(
            [
                uniq_scene_name_to_label[scene_name]
                for scene_name in dataset_df["scene"].values
            ],
            dtype=np.int64,
        )

        image_paths = dataset_df.apply(data.resolve_image_path, axis=1)
        y = clustering.run(list(image_paths.values)).cluster_labels.copy()

        ami = sklearn.metrics.adjusted_mutual_info_score(t, y)
        print(f"dataset={dataset_name}: #clusters={len(set(y.tolist()))} AMI={ami}")
        scores.append(ami)

    print(np.array(scores).mean())


if __name__ == "__main__":
    main()
