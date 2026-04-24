from typing import Callable, List, Optional, Tuple, Union

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel

from scripts.data import FilePath, LocalFeatureOutputs, resolve_model_path
from features.base import LocalFeatureHandler, read_image
from features.config import FeatureSetConfig
from postprocesses.nms import nms_local_features
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper


class FeatureSetHandler(LocalFeatureHandler):
    def __init__(
        self,
        handlers: List[LocalFeatureHandler],
        conf: FeatureSetConfig,
        device: Optional[torch.device] = None,
    ):
        self.handlers = handlers
        self.conf = conf
        self.device = device

    @torch.inference_mode()
    def __call__(
        self,
        path: FilePath,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image,
    ) -> LocalFeatureOutputs:
        assert resize is None
        assert rotation is None

        lafs_list = []
        scores_list = []
        descs_list = []

        for handler, c in zip(self.handlers, self.conf.local_features):
            if c.ignore_cropper:
                _cropper = None
            else:
                _cropper = cropper
            lafs, scores, descs = handler(
                path,
                resize=c.resize,
                rotation=c.rotation,
                cropper=_cropper,
                orientation=orientation,
                image_reader=image_reader,
            )
            lafs_list.append(lafs)
            scores_list.append(scores)
            descs_list.append(descs)

        lafs = torch.cat(lafs_list)
        scores = torch.cat(scores_list)
        descs = torch.cat(descs_list)

        if self.conf.nms:
            img = image_reader(str(path))
            lafs, scores, descs = nms_local_features(
                lafs, scores, descs, img, self.conf.nms
            )

        if self.conf.topk:
            order = -scores.argsort()
            order = order[: self.conf.topk]
            lafs = lafs[order]
            scores = scores[order]
            descs = descs[order]

        return lafs, scores, descs

    @torch.inference_mode()
    def extract_by_keypoints(
        self,
        path: FilePath,
        pre_sampled_keypoints: np.ndarray,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image,
    ) -> LocalFeatureOutputs:
        lafs_list = []
        scores_list = []
        descs_list = []

        for handler, c in zip(self.handlers, self.conf.local_features):
            if c.ignore_cropper:
                _cropper = None
            else:
                _cropper = cropper
            lafs, scores, descs = handler.extract_by_keypoints(
                path,
                pre_sampled_keypoints,
                resize=c.resize,
                rotation=c.rotation,
                cropper=_cropper,
                orientation=orientation,
                image_reader=image_reader,
            )
            lafs_list.append(lafs)
            scores_list.append(scores)
            descs_list.append(descs)

        lafs = torch.cat(lafs_list)
        scores = torch.cat(scores_list)
        descs = torch.cat(descs_list)

        if self.conf.nms:
            img = image_reader(str(path))
            lafs, scores, descs = nms_local_features(
                lafs, scores, descs, img, self.conf.nms
            )

        if self.conf.topk:
            order = -scores.argsort()
            order = order[: self.conf.topk]
            lafs = lafs[order]
            scores = scores[order]
            descs = descs[order]

        return lafs, scores, descs
