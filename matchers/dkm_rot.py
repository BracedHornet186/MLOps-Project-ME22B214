from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.cuda.amp
from PIL import Image

from data import FilePath, resolve_model_path
from matchers.base import DetectorFreeMatcher
from matchers.config import DKMRotationConfig
from models.dkm.models.model_zoo import DKMv3_outdoor
from postprocesses.nms import nms_matched_keypoints, sort_matched_keypoints_by_score
from preprocess import resize_image_opencv
from preprocesses.homography_adaptation import HomographyAdaptation
from preprocesses.region import OverlapRegionCropper
from storage import MatchedKeypointStorage
from workspace import log


def read_image(path: str) -> np.ndarray:
    return cv2.imread(str(path))


class DKMRotationMatcher(DetectorFreeMatcher):
    def __init__(self, conf: DKMRotationConfig, device: Optional[torch.device] = None):
        weight_path = resolve_model_path(conf.dkm.weight_path)
        if conf.dkm.model_type == "DKMv3_outdoor":
            model = DKMv3_outdoor(
                path_to_weights=str(weight_path),
                height=conf.dkm.height,
                width=conf.dkm.width,
                upsample_preds=conf.dkm.upsample_preds,
                sample_mode=conf.dkm.sample_mode,
                device=torch.device("cpu"),
            )
        else:
            raise ValueError(conf.dkm.model_type)

        if conf.dkm.sample_threshold is not None:
            model.sample_thresh = conf.dkm.sample_threshold
        log(f"[DKMRotationMatcher] Weights were loaded from {weight_path}")
        log(f"[DKMRotationMatcher] Use the device ({device})")
        log(
            f"[DKMRotationMatcher] DKM.height={model.h_resized}, "
            f"DKM.width={model.w_resized}, "
            f"DKM.upsample_preds={model.upsample_preds}, "
            f"DKM.sample_mode={model.sample_mode}, "
            f"DKM.sample_thresh={model.sample_thresh}"
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
        assert self.conf.pre_resize

        orig_img1 = image_reader(str(path1))
        orig_img2 = image_reader(str(path2))

        orig_img1 = cv2.cvtColor(orig_img1, cv2.COLOR_BGR2RGB)
        orig_img2 = cv2.cvtColor(orig_img2, cv2.COLOR_BGR2RGB)

        resized_img1, scale1, mask1 = resize_image_opencv(
            orig_img1, self.conf.pre_resize, order3ch="hwc"
        )
        resized_img2, scale2, mask1 = resize_image_opencv(
            orig_img1, self.conf.pre_resize, order3ch="hwc"
        )

        mkpts1_list = []
        mkpts2_list = []
        scores_list = []
        num_matches = []
        certanty_sum_list = []

        image1 = Image.fromarray(resized_img1)
        for angle in self.conf.angles:
            _height, _width = resized_img2.shape[:2]

            rotator = HomographyAdaptation(_height, _width, [angle])

            rotated_img2 = rotator.transform_homography_variants(resized_img2)[0]
            image2 = Image.fromarray(rotated_img2)

            matches, certainties = self.model.match(image1, image2, device=self.device)
            num_matches.append(matches.shape[0])
            certanty_sum_list.append(certainties.sum().item())

            matches, certainties = self.model.sample(
                matches, certainties, num=self.conf.dkm.sample_nums
            )
            W1, H1 = image1.size
            W2, H2 = image2.size
            mkpts1, mkpts2 = self.model.to_pixel_coordinates(matches, H1, W1, H2, W2)
            mkpts1 = rotator.inverse_transform_keypoints_tensor(mkpts1)
            mkpts2 = rotator.inverse_transform_keypoints_tensor(mkpts2)

            mkpts1 = mkpts1.cpu().numpy()
            mkpts2 = mkpts2.cpu().numpy()
            scores = certainties.cpu().numpy()

            mkpts1_list.append(mkpts1)
            mkpts2_list.append(mkpts2)
            scores_list.append(scores)

        if False:
            # Concat
            mkpts1 = np.concatenate(mkpts1_list, axis=0)
            mkpts2 = np.concatenate(mkpts2_list, axis=0)
            scores = np.concatenate(scores_list, axis=0)
        else:
            if len(num_matches) == 0:
                return
            if sum(num_matches) == 0:
                return

            best_matching_idx = int(np.array(certanty_sum_list).argsort()[-1])
            print(certanty_sum_list, best_matching_idx)
            mkpts1 = mkpts1_list[best_matching_idx]
            mkpts2 = mkpts2_list[best_matching_idx]
            scores = scores_list[best_matching_idx]

        mkpts1[:, 0] *= scale1[0]
        mkpts1[:, 1] *= scale1[1]
        mkpts2[:, 0] *= scale2[0]
        mkpts2[:, 1] *= scale2[1]

        if self.conf.nms:
            mkpts1, mkpts2, scores = sort_matched_keypoints_by_score(
                mkpts1, mkpts2, scores
            )
            mkpts1, mkpts2, scores = nms_matched_keypoints(
                mkpts1, mkpts2, scores, self.conf.nms, image_shape=orig_img1.shape[:2]
            )

        if self.conf.min_matches is None or len(mkpts1) >= self.conf.min_matches:
            matched_keypoint_storage.add(path1, path2, mkpts1, mkpts2, scores=scores)
