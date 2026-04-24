from typing import Optional

import cv2
import kornia
import matplotlib.figure
import matplotlib.pyplot as plt
import kornia
from kornia_moons.feature import visualize_LAF

from scripts.data import FilePath
from features.base import LocalFeatureHandler, LocalFeatureOutputs
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class LocalFeatureVisualizer:
    def __init__(self, handler: LocalFeatureHandler):
        self.handler = handler
    
    def __call__(
        self,
        path: FilePath,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        return_fig: bool = False,
        print_keypoints: bool = False
    ) -> Optional[matplotlib.figure.Figure]:
        """Show extracted keypoints
        """
        print(f'[LocalFeatureVisualizer] {path}')
        lafs, scores, descs = self.handler(path, resize=resize, rotation=rotation)

        if print_keypoints:
            kpts = kornia.feature.get_laf_center(lafs[None])[0]
            print(kpts)
            print(kpts.shape)

        img = kornia.utils.image.image_to_tensor(
            cv2.imread(str(path)), keepdim=False).float() / 255.0
        img = kornia.color.bgr_to_rgb(img)

        fig = visualize_LAF(img, lafs[None, ...])
        if return_fig:
            return fig
