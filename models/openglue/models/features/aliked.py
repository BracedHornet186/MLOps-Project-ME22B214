import pathlib
from typing import Optional, Tuple, Union
from matplotlib.colors import to_rgb

import kornia
import cv2
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F

from features.aliked import ALIKEDHandler
from features.config import ALIKEDConfig, LocalFeatureConfig
from features.factory import create_local_feature_handler
from preprocesses.config import ResizeConfig
from scripts.preprocess import resize_image_opencv
from scripts.extractor import LocalFeatureExtractor
from scripts.data import FilePath


CONFIG_V1 = LocalFeatureConfig(
    type='aliked',
    resize=ResizeConfig(
        func='opencv-divisible',
        long_edge_length=1600,
        antialias=True
    ),
    aliked=ALIKEDConfig(
        weight_path='/home/ns64/git_repos/kaggle-image-matching-challenge-2023/extra/pretrained_models/aliked/aliked-n32.pth',
        top_k=-1,
        scores_th=0.2,
        n_limit=4096
    )
)


def load_model(
    conf: LocalFeatureConfig
) -> ALIKEDHandler:
    device = torch.device('cuda')
    handler = create_local_feature_handler(conf, device=device)
    return handler


def read_image(
    path: FilePath,
    resize: Optional[ResizeConfig] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize:
        resized_img, scale, mask = resize_image_opencv(img, conf=resize)
    else:
        raise NotImplementedError

    return resized_img, img, scale


class ALIKEDFeature(nn.Module):
    def __init__(self, load_path: str):
        super().__init__()

        conf = CONFIG_V1
        print(conf)
        #conf.weight_path = load_path

        self.conf = conf
        self.model = load_model(conf)
        self.device = torch.device('cuda:0')

    def read_image(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return read_image(path, resize=self.conf.resize)

    def forward(self, resized_img: np.ndarray, img: np.ndarray, scale: np.ndarray, mask=None):
        """Compute keypoints, scores, descriptors for image"""
        preds = self.model.model.run(resized_img)

        kpts = torch.from_numpy(preds['keypoints']).to(self.device, non_blocking=True)
        scores = torch.from_numpy(preds['scores']).to(self.device, non_blocking=True)
        descs = torch.from_numpy(preds['descriptors']).to(self.device, non_blocking=True)

        lafs = kornia.feature.laf_from_center_scale_ori(
            kpts[None], torch.ones(1, len(kpts), 1, 1, device=self.device)
        )[0]    # Shape(1, N, 2, 3) -> Shape(N, 2, 3)

        outputs = (lafs, scores, descs)
        outputs = self.model.postprocess(
            outputs, img,
            kornia.utils.image_to_tensor(resized_img, keepdim=False)
        )
        return outputs