from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.cuda.amp
from PIL import Image

from scripts.data import FilePath, resolve_model_path
from matchers.base import DetectorFreeMatcher
from matchers.config import RoMaConfig
from models.roma.models.model_zoo import roma_outdoor
from postprocesses.nms import nms_matched_keypoints, sort_matched_keypoints_by_score
from preprocesses.orientation import OrientationNormalizer
from preprocesses.region import OverlapRegionCropper
from scripts.storage import MatchedKeypointStorage
from scripts.workspace import log


def read_image(path: str) -> np.ndarray:
    return cv2.imread(str(path))


class RoMaMatcher(DetectorFreeMatcher):
    def __init__(self, conf: RoMaConfig, device: Optional[torch.device] = None):
        assert device is not None

        weight_path = resolve_model_path(conf.weight_path)
        dinov2_weight_path = resolve_model_path(conf.dinov2_weight_path)
        if conf.model_type == "outdoor":
            weights = torch.load(weight_path, map_location="cpu")
            dinov2_weights = torch.load(dinov2_weight_path, map_location="cpu")
            model = roma_outdoor(
                device,
                weights=weights,
                dinov2_weights=dinov2_weights,
                coarse_res=conf.coarse_res,
                upsample_res=conf.upsample_res,
            )
        else:
            raise ValueError(conf.model_type)

        if not conf.upsample_preds:
            model.upsample_preds = False
        
        if conf.sample_threshold is not None:
            model.sample_thresh = float(conf.sample_threshold)

        log(f"[RoMaMatcher] Weights were loaded from {weight_path}")
        log(f"[RoMaMatcher] Use the device ({device})")
        log(
            f"[RoMaMatcher] RoMa.height={model.h_resized}, "
            f"RoMa.width={model.w_resized}, "
            f"RoMa.upsample_preds={model.upsample_preds}, "
            f"RoMa.sample_mode={model.sample_mode}, "
            f"RoMa.sample_thresh={model.sample_thresh}"
        )
        self.model = model.eval().to(device)
        self.conf = conf
        self.device = device

    @torch.inference_mode()
    def __call__(
        self,
        path1: FilePath,
        path2: FilePath,
        matched_keypoint_storage: MatchedKeypointStorage,
        cropper: Optional[OverlapRegionCropper] = None,
        orientation1: Optional[int] = None,
        orientation2: Optional[int] = None,
        image_reader: Callable = read_image,
    ):
        img1 = image_reader(str(path1))
        img2 = image_reader(str(path2))

        ori_normalizer1 = OrientationNormalizer.create_if_needed(orientation1)
        ori_normalizer2 = OrientationNormalizer.create_if_needed(orientation2)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        if cropper:
            cropper.set_original_image(img1, img2)
            img1, img2 = cropper.crop_ndarray_image(img1, img2)

        if ori_normalizer1:
            ori_normalizer1.set_original_image(img1)
            img1 = ori_normalizer1.get_upright_image_ndarray()
        if ori_normalizer2:
            ori_normalizer2.set_original_image(img2)
            img2 = ori_normalizer2.get_upright_image_ndarray()

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        warp, certainties = self.model.match(img1, img2, device=self.device)
        matches, certainties = self.model.sample(
            warp, certainties, num=self.conf.sample_nums
        )
        W1, H1 = img1.size
        W2, H2 = img2.size
        mkpts1, mkpts2 = self.model.to_pixel_coordinates(matches, H1, W1, H2, W2)
        mkpts1 = mkpts1.cpu().numpy()
        mkpts2 = mkpts2.cpu().numpy()
        scores = certainties.cpu().numpy()

        mkpts1, mkpts2, scores = sort_matched_keypoints_by_score(
            mkpts1, mkpts2, scores
        )
        if self.conf.nms:
            mkpts1, mkpts2, scores = nms_matched_keypoints(
                mkpts1, mkpts2, scores, self.conf.nms, img=np.array(img1)
            )

        if self.conf.confidence_threshold is not None:
            keep = scores >= self.conf.confidence_threshold
            mkpts1 = mkpts1[keep]
            mkpts2 = mkpts2[keep]
            scores = scores[keep]

        if self.conf.topk is not None:
            k = self.conf.topk
            mkpts1 = mkpts1[:k]
            mkpts2 = mkpts2[:k]
            scores = scores[:k]

        if ori_normalizer1:
            mkpts1 = ori_normalizer1.keypoints_to_original_coords_ndarray(mkpts1)
        if ori_normalizer2:
            mkpts2 = ori_normalizer2.keypoints_to_original_coords_ndarray(mkpts2)

        if cropper:
            mkpts1, mkpts2 = cropper.convert_cropped_to_original_coordinates(
                mkpts1, mkpts2
            )

        if self.conf.min_matches is None or len(mkpts1) >= self.conf.min_matches:
            matched_keypoint_storage.add(path1, path2, mkpts1, mkpts2, scores=scores)

    @torch.inference_mode()
    def match_keypoints(
        self,
        path1: FilePath,
        path2: FilePath,
        kpts1: np.ndarray,
        matched_keypoint_storage: MatchedKeypointStorage,
        cropper: Optional[OverlapRegionCropper] = None,
        image_reader: Callable = read_image,
    ):
        img1 = image_reader(str(path1))
        img2 = image_reader(str(path2))

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        if cropper:
            cropper.set_original_image(img1, img2)
            img1, img2 = cropper.crop_ndarray_image(img1, img2)

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        warp, certainties = self.model.match(img1, img2, device=self.device)

        self.model.match_keypoints(
        )
        matches, certainties = self.model.sample(
            warp, certainties, num=self.conf.sample_nums
        )
        W1, H1 = img1.size
        W2, H2 = img2.size
        mkpts1, mkpts2 = self.model.to_pixel_coordinates(matches, H1, W1, H2, W2)
        mkpts1 = mkpts1.cpu().numpy()
        mkpts2 = mkpts2.cpu().numpy()
        scores = certainties.cpu().numpy()

        mkpts1, mkpts2, scores = sort_matched_keypoints_by_score(
            mkpts1, mkpts2, scores
        )
        if self.conf.nms:
            mkpts1, mkpts2, scores = nms_matched_keypoints(
                mkpts1, mkpts2, scores, self.conf.nms, img=np.array(img1)
            )

        if self.conf.confidence_threshold is not None:
            keep = scores >= self.conf.confidence_threshold
            mkpts1 = mkpts1[keep]
            mkpts2 = mkpts2[keep]
            scores = scores[keep]

        if self.conf.topk is not None:
            k = self.conf.topk
            mkpts1 = mkpts1[:k]
            mkpts2 = mkpts2[:k]
            scores = scores[:k]

        if ori_normalizer1:
            mkpts1 = ori_normalizer1.keypoints_to_original_coords_ndarray(mkpts1)
        if ori_normalizer2:
            mkpts2 = ori_normalizer2.keypoints_to_original_coords_ndarray(mkpts2)

        if cropper:
            mkpts1, mkpts2 = cropper.convert_cropped_to_original_coordinates(
                mkpts1, mkpts2
            )

        if self.conf.min_matches is None or len(mkpts1) >= self.conf.min_matches:
            matched_keypoint_storage.add(path1, path2, mkpts1, mkpts2, scores=scores)