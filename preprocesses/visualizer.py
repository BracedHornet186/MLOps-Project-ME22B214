from typing import Optional

import cv2
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch
import pycolmap
from kornia_moons.feature import draw_LAF_matches
from PIL import Image

from data import FilePath
from extractor import LocalFeatureExtractor
from features.base import read_image
from matchers.base import DetectorFreeMatcher, LocalFeatureMatcher
from storage import (InMemoryKeypointStorage, InMemoryLocalFeatureStorage,
                     InMemoryMatchedKeypointStorage, InMemoryMatchingStorage)
from preprocesses.region import OverlapRegionEstimatorConfig, OverlapRegionEstimator

plt.rcParams['figure.dpi'] = 200


class OverlapEstimatorVisualizer:
    def __init__(self,
                 matcher: DetectorFreeMatcher,
                 estimator: OverlapRegionEstimator):
        self.matcher = matcher
        self.estimator = estimator

    def __call__(self, path1: FilePath, path2: FilePath):
        m_storage = InMemoryMatchingStorage()
        f_storage = InMemoryKeypointStorage()
        mkpt_storage = InMemoryMatchedKeypointStorage()

        img1 = cv2.imread(str(path1))
        img2 = cv2.imread(str(path2))
        self.matcher(path1, path2, mkpt_storage, image_reader=read_image)

        mkpts1, mkpts2 = mkpt_storage.get(path1, path2)
        print(f'(path1, path2) = ({path1}, {path2})')
        print(f'#matches = {len(mkpts1)}')

        shape1 = (img1.shape[0], img1.shape[1])
        shape2 = (img2.shape[0], img2.shape[1])
        bboxes1, bboxes2 = self.estimator.get_paired_bboxes(
            mkpts1, mkpts2, shape1, shape2
        )
        print(bboxes1, bboxes2)

        draw(path1, path2, mkpts1, mkpts2, bbox1=bboxes1[0], bbox2=bboxes2[0])


def draw(path1: FilePath,
         path2: FilePath,
         mkpts1: np.ndarray,
         mkpts2: np.ndarray,
         inliers: Optional[np.ndarray] = None,
         bbox1: Optional[np.ndarray] = None,
         bbox2: Optional[np.ndarray] = None,
         ax: Optional[plt.Axes] = None):
    if ax is None:
        fig, ax = plt.subplots()

    draw_dict = {'inlier_color': None,
                 'tentative_color': None, 
                 'feature_color': None,
                 'vertical': False}
    if False:
        draw_dict['inlier_color'] = (0.2, 1, 0.2)
        draw_dict['tentative_color'] = (1, 0.2, 0.2)
    draw_dict['feature_color'] = (0.2, 0.5, 1)

    pil_image1 = Image.open(path1).convert('RGB')
    pil_image2 = Image.open(path2).convert('RGB')
    image1 = np.array(pil_image1)
    image2 = np.array(pil_image2)
    if inliers is None:
        inliers = np.ones((len(mkpts1),)).astype(bool)

    if bbox1 is not None:
        image1 = cv2.rectangle(
            image1,
            pt1=(int(bbox1[0]), int(bbox1[1])),
            pt2=(int(bbox1[2]), int(bbox1[3])),
            color=(0, 255, 0),
            thickness=3
        )
    if bbox2 is not None:
        image2 = cv2.rectangle(
            image2,
            pt1=(int(bbox2[0]), int(bbox2[1])),
            pt2=(int(bbox2[2]), int(bbox2[3])),
            color=(0, 255, 0),
            thickness=3
        )

    draw_LAF_matches(
        kornia.feature.laf_from_center_scale_ori(
            torch.from_numpy(mkpts1).view(1,-1, 2),
            torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
            torch.ones(mkpts1.shape[0]).view(1,-1, 1)
        ),
        kornia.feature.laf_from_center_scale_ori(
            torch.from_numpy(mkpts2).view(1,-1, 2),
            torch.ones(mkpts2.shape[0]).view(1,-1, 1, 1),
            torch.ones(mkpts2.shape[0]).view(1,-1, 1)
        ),
        torch.arange(mkpts1.shape[0]).view(-1,1).repeat(1,2),
        image1,
        image2,
        inliers,
        draw_dict=draw_dict,
        ax=ax
    )