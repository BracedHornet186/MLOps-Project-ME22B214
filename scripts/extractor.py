from typing import Any, Callable, Optional

import numpy as np
import tqdm

from scripts.data import FilePath, LocalFeatureExtractionOutputs
from features.base import (
    Line2DFeatureHandler,
    LocalFeatureHandler,
    lafs_to_keypoints,
    read_image,
)
from features.config import Line2DFeatureConfig, LocalFeatureConfig
from pipelines.scene import Scene
from preprocesses.config import ResizeConfig, RotationConfig
from preprocesses.region import Cropper
from scripts.storage import (
    Line2DFeatureStorage,
    Line2DSegmentStorage,
    LocalFeatureStorage,
    InMemoryLine2DFeatureStorage,
    filter_local_features_by_mask_regions,
)


class LocalFeatureExtractor:
    def __init__(
        self,
        conf: LocalFeatureConfig,
        handler: LocalFeatureHandler,
    ):
        self.conf = conf
        self.handler = handler

    def __call__(
        self,
        path: FilePath,
        cropper: Optional[Cropper] = None,
        rotation: Optional[RotationConfig] = None,
        orientation: Optional[int] = None,
        pre_sampled_keypoints: Optional[np.ndarray] = None,
        image_reader: Callable = read_image,
        storage: Optional[LocalFeatureStorage] = None,  # Out
    ) -> LocalFeatureExtractionOutputs:
        rotation = rotation or self.conf.rotation
        outputs = self.extract(
            path,
            resize=self.conf.resize,
            rotation=rotation,
            cropper=cropper,
            orientation=orientation,
            pre_sampled_keypoints=pre_sampled_keypoints,
            image_reader=image_reader,
        )
        if storage:
            storage.add(path, outputs)
        return outputs

    def extract(
        self,
        path: FilePath,
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        pre_sampled_keypoints: Optional[np.ndarray] = None,
        image_reader: Callable = read_image,
    ) -> LocalFeatureExtractionOutputs:
        if self.conf.ignore_cropper:
            cropper = None

        if self.conf.extract_from_pre_sampled_keypoints:
            if pre_sampled_keypoints is None:
                print("pre_sampled_keypoints are required, but None")
            elif len(pre_sampled_keypoints) == 0:
                print("pre_sampled_keypoints have 0 points")
            else:
                try:
                    lafs, scores, descs = self.handler.extract_by_keypoints(
                        path,
                        pre_sampled_keypoints,
                        resize=resize,
                        rotation=rotation,
                        cropper=cropper,
                        orientation=orientation,
                        image_reader=image_reader,
                    )
                    kpts = lafs_to_keypoints(lafs)
                    return (
                        lafs.detach().cpu().float().numpy(),
                        kpts.detach().cpu().float().numpy(),
                        scores.detach().cpu().float().numpy(),
                        descs.detach().cpu().float().numpy(),
                    )
                except Exception as e:
                    print(f"Error extract_by_keypoints: {e}")

        lafs, scores, descs = self.handler(
            path,
            resize=resize,
            rotation=rotation,
            cropper=cropper,
            orientation=orientation,
            image_reader=image_reader,
        )

        kpts = lafs_to_keypoints(lafs)
        return (
            lafs.detach().cpu().float().numpy(),
            kpts.detach().cpu().float().numpy(),
            scores.detach().cpu().float().numpy(),
            descs.detach().cpu().float().numpy(),
        )


class Line2DFeatureExtractor:
    def __init__(
        self,
        conf: Line2DFeatureConfig,
        handler: Line2DFeatureHandler,
    ):
        self.conf = conf
        self.handler = handler

    def __call__(
        self,
        path: FilePath,
        shape: tuple[int, int],
        cropper: Optional[Cropper] = None,
        rotation: Optional[RotationConfig] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image,
        storage: Optional[Line2DFeatureStorage] = None,
    ) -> tuple[np.ndarray, Any]:
        segs, descinfos = self.extract(
            path,
            shape,
            resize=self.conf.resize,
            rotation=rotation,
            cropper=cropper,
            orientation=orientation,
            image_reader=image_reader,
        )
        if storage:
            storage.add(path, segs, descinfos)
        return segs, descinfos

    def extract(
        self,
        path: FilePath,
        shape: tuple[int, int],
        resize: Optional[ResizeConfig] = None,
        rotation: Optional[RotationConfig] = None,
        cropper: Optional[Cropper] = None,
        orientation: Optional[int] = None,
        image_reader: Callable = read_image,
    ) -> tuple[np.ndarray, Any]:
        segs, descinfos = self.handler(
            path,
            shape,
            resize=resize,
            image_reader=image_reader,
        )
        return segs, descinfos


def extract_all(
    extractor: LocalFeatureExtractor,
    scene: Scene,
    line2d_seg_storage: Optional[Line2DSegmentStorage] = None,
    storage: Optional[LocalFeatureStorage] = None,  # Out
    progress_bar: Optional[tqdm.tqdm] = None,
) -> None:
    paths = scene.image_paths
    for i, path in enumerate(paths):
        path = str(path)

        cropper = scene.create_roi_region_cropper_if_exists(path)
        orientation = scene.get_orientation_degree(path)

        pre_sampled_keypoints = None
        if line2d_seg_storage:
            if extractor.conf.pre_sampled_keypoints_interpolation_num is None:
                pre_sampled_keypoints = line2d_seg_storage.get_endpoints(path)
            else:
                num = extractor.conf.pre_sampled_keypoints_interpolation_num
                assert num > 0
                pre_sampled_keypoints = (
                    line2d_seg_storage.get_endpoints_with_interpolation(path, n=num)
                )

        extractor(
            path,
            cropper=cropper,
            orientation=orientation,
            pre_sampled_keypoints=pre_sampled_keypoints,
            storage=storage,
            image_reader=scene.get_image,
        )

        if storage and scene.mask_bboxes:
            print("Filtering local features based on mask regions")
            storage = filter_local_features_by_mask_regions(storage, scene)

        if progress_bar:
            progress_bar.set_postfix_str(
                f"Extract local features ({i + 1}/{len(paths)})"
            )


def extract_line2d_features_all(
    extractor: Line2DFeatureExtractor,
    scene: Scene,
    storage: Optional[Line2DFeatureStorage] = None,  # out
    progress_bar: Optional[tqdm.tqdm] = None,
) -> Line2DFeatureStorage:
    storage = storage or InMemoryLine2DFeatureStorage()

    paths = scene.image_paths
    for i, path in enumerate(paths):
        path = str(path)

        shape = scene.get_image_shape(path)

        cropper = None
        bbox = scene.bboxes.get(path)
        if bbox is not None:
            cropper = Cropper(bbox, scene.get_image_shape(path))

        orientation = scene.get_orientation_degree(path)

        try:
            extractor(
                path,
                shape,
                cropper=cropper,
                orientation=orientation,
                storage=storage,
                image_reader=scene.get_image,
            )
        except Exception as e:
            print(f"Line2D extraction error: {e}, {path}")

        if storage and scene.mask_bboxes:
            # TODO
            pass

        if progress_bar:
            progress_bar.set_postfix_str(
                f"Extract line2d features ({i + 1}/{len(paths)})"
            )

    return storage
