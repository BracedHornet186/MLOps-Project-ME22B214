from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid

from clusterings.base import Clustering, ClusteringResult


class ClusteringVisualizer:
    def __init__(self, clustering: Clustering):
        self.clustering = clustering

    def visualize(self, result: ClusteringResult):
        image_paths = result.image_paths
        cluster_labels = result.cluster_labels
        unique_labels = np.unique(cluster_labels)

        fig, axs = plt.subplots(len(unique_labels), 1)
        for a, label in enumerate(unique_labels):
            indices, *_ = np.where(cluster_labels == label)
            cluster_image_paths = [image_paths[i] for i in indices]
            cluster_img = make_list_image(cluster_image_paths)
            if len(unique_labels) == 1:
                axs.imshow(cluster_img)
                axs.set_title(f"label={label}")
            else:
                axs[a].imshow(cluster_img)
                axs[a].set_title(f"label={label}")


def make_list_image(
    image_paths: list[str | Path],
    cell_size: int = 64,
    max_cols: int = 20,
):
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((cell_size, cell_size))
        img = pad2square(img)
        images.append(img)
    tensor_img = torch.stack(
        [torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in images]
        + [
            torch.zeros((3, cell_size, cell_size), dtype=torch.uint8)
            for _ in range(max_cols - len(images) % max_cols)
        ]
    )
    grid_img_tensor = make_grid(tensor_img, nrow=max_cols)
    return Image.fromarray(grid_img_tensor.cpu().numpy().transpose(1, 2, 0))


def pad2square(img: Image.Image) -> Image.Image:
    width = img.width
    height = img.height
    if width == height:
        return img
    elif width > height:
        result = Image.new(img.mode, (width, width), (0, 0, 0))
        result.paste(img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(img.mode, (height, height), (0, 0, 0))
        result.paste(img, ((height - width) // 2, 0))
        return result
