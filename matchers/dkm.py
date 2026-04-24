from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.cuda.amp
from PIL import Image

from scripts.data import FilePath, resolve_model_path
from matchers.base import DetectorFreeMatcher
from matchers.config import DKMConfig
from models.dkm.models.model_zoo import DKMv3_outdoor
from pipelines.verification import run_ransac
from postprocesses.nms import nms_matched_keypoints, sort_matched_keypoints_by_score
from postprocesses.panet import PANetRefiner
from preprocesses.orientation import OrientationNormalizer
from preprocesses.region import OverlapRegionCropper
from scripts.storage import MatchedKeypointStorage
from scripts.workspace import log


def read_image(path: str) -> np.ndarray:
    return cv2.imread(str(path))


class DKMMatcher(DetectorFreeMatcher):
    def __init__(self, conf: DKMConfig, device: Optional[torch.device] = None):
        weight_path = resolve_model_path(conf.weight_path)
        if conf.model_type == "DKMv3_outdoor":
            model = DKMv3_outdoor(
                path_to_weights=str(weight_path),
                height=conf.height,
                width=conf.width,
                upsample_preds=conf.upsample_preds,
                sample_mode=conf.sample_mode,
                device=torch.device("cpu"),
            )
        else:
            raise ValueError(conf.model_type)

        if conf.sample_threshold is not None:
            model.sample_thresh = conf.sample_threshold
        if conf.upsample_height is not None:
            assert conf.upsample_width is not None
            model.upsample_res = (int(conf.upsample_height), int(conf.upsample_width))
        log(f"[DKMMatcher] Weights were loaded from {weight_path}")
        log(f"[DKMMatcher] Use the device ({device})")
        log(
            f"[DKMMatcher] DKM.height={model.h_resized}, "
            f"DKM.width={model.w_resized}, "
            f"DKM.upsample_preds={model.upsample_preds}, "
            f"DKM.upsample_res={model.upsample_res}, "
            f"DKM.sample_mode={model.sample_mode}, "
            f"DKM.sample_thresh={model.sample_thresh}"
        )
        self.model = model.eval().to(device)
        self.conf = conf
        self.device = device

        refiner = None
        if conf.panet:
            refiner = PANetRefiner(conf.panet, device)
            print(f"[DKMMatcher] Use refiner: {refiner}")
        self.refiner = refiner

        if conf.verification:
            print("[DKMMatcher] Use pre-verification")

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
        orig_img1 = image_reader(str(path1))
        orig_img2 = image_reader(str(path2))

        ori_normalizer1 = OrientationNormalizer.create_if_needed(orientation1)
        ori_normalizer2 = OrientationNormalizer.create_if_needed(orientation2)

        orig_img1 = cv2.cvtColor(orig_img1, cv2.COLOR_BGR2RGB)
        orig_img2 = cv2.cvtColor(orig_img2, cv2.COLOR_BGR2RGB)

        if cropper:
            cropper.set_original_image(orig_img1, orig_img2)
            img1, img2 = cropper.crop_ndarray_image(orig_img1, orig_img2)
        else:
            img1 = orig_img1
            img2 = orig_img2

        if ori_normalizer1:
            ori_normalizer1.set_original_image(img1)
            img1 = ori_normalizer1.get_upright_image_ndarray()
        if ori_normalizer2:
            ori_normalizer2.set_original_image(img2)
            img2 = ori_normalizer2.get_upright_image_ndarray()

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        matches, certainties = self.model.match(img1, img2, device=self.device)
        if self.conf.confidence_threshold is not None:
            keep = certainties >= self.conf.confidence_threshold
            matches = matches[keep]
            certainties = certainties[keep]

        if len(matches) == 0:
            return

        matches, certainties = self.model.sample(
            matches, certainties, num=self.conf.sample_nums
        )
        W1, H1 = img1.size
        W2, H2 = img2.size
        mkpts1, mkpts2 = self.model.to_pixel_coordinates(matches, H1, W1, H2, W2)
        mkpts1 = mkpts1.cpu().numpy()
        mkpts2 = mkpts2.cpu().numpy()
        scores = certainties.cpu().numpy()

        mkpts1, mkpts2, scores = sort_matched_keypoints_by_score(mkpts1, mkpts2, scores)
        if self.conf.topk is not None:
            mkpts1 = mkpts1[: self.conf.topk]
            mkpts2 = mkpts2[: self.conf.topk]
            scores = scores[: self.conf.topk]

        if self.conf.nms:
            mkpts1, mkpts2, scores = nms_matched_keypoints(
                mkpts1, mkpts2, scores, self.conf.nms, img=np.array(img1)
            )

        if self.refiner is not None:
            mkpts1, mkpts2 = self.refiner.refine_matched_keypoints(
                np.array(img1),
                np.array(img2),
                mkpts1,
                mkpts2,
                np.stack([np.arange(len(mkpts1)), np.arange(len(mkpts1))]).T,
            )

        if ori_normalizer1:
            mkpts1 = ori_normalizer1.keypoints_to_original_coords_ndarray(mkpts1)
        if ori_normalizer2:
            mkpts2 = ori_normalizer2.keypoints_to_original_coords_ndarray(mkpts2)

        if cropper:
            mkpts1, mkpts2 = cropper.convert_cropped_to_original_coordinates(
                mkpts1, mkpts2
            )

        if self.conf.min_matches is None or len(mkpts1) >= self.conf.min_matches:
            if self.conf.verification:
                min_matches = self.conf.min_matches or 16
                _, inliers = run_ransac(mkpts1, mkpts2, self.conf.verification, min_matches_required=min_matches)
                inliers = (inliers > 0).reshape(-1)
                mkpts1 = mkpts1[inliers]
                mkpts2 = mkpts2[inliers]
                scores = scores[inliers]
            matched_keypoint_storage.add(path1, path2, mkpts1, mkpts2, scores=scores)
