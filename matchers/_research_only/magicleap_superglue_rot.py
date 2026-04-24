from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch

from scripts.data import FilePath, resolve_model_path
from scripts.extractor import LocalFeatureExtractor
from features.factory import create_local_feature_handler
from matchers.base import DetectorFreeMatcher
from matchers.config import MagicLeapSuperGlueRotationConfig
from models._research_only.magicleap.superglue import SuperGlue
from preprocesses.config import RotationConfig
from preprocesses.region import OverlapRegionCropper
from scripts.storage import MatchedKeypointStorage


def read_image(path: str) -> np.ndarray:
    return cv2.imread(str(path))


class CachedImageReader:
    def __init__(self, img: np.ndarray):
        self.img = img

    def __call__(self, path: str) -> np.ndarray:
        return self.img.copy()


class MagicLeapSuperGlueRotationMatcher(DetectorFreeMatcher):
    def __init__(
        self,
        conf: MagicLeapSuperGlueRotationConfig,
        device: Optional[torch.device] = None,
    ):
        weight_path = str(resolve_model_path(conf.ml_superglue.weight_path))
        model = SuperGlue(
            {
                "weights": conf.ml_superglue.weights,
                "sinkhorn_iterations": conf.ml_superglue.sinkhorn_iterations,
                "match_threshold": conf.ml_superglue.match_threshold,
                "model_path": weight_path,
            }
        )

        assert conf.local_feature.type == "magicleap_superpoint"
        handler = create_local_feature_handler(conf.local_feature, device=device)
        extractor = LocalFeatureExtractor(conf.local_feature, handler)

        self.conf = conf
        self.device = device
        self.model = model.eval().to(device)
        self.extractor = extractor

    @property
    def min_matches(self) -> Optional[int]:
        return self.conf.min_matches

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
        img1, img2 = image_reader(str(path1)), image_reader(str(path2))
        shape1 = img1.shape[:2]
        shape2 = img2.shape[:2]

        image_reader1 = CachedImageReader(img1)
        image_reader2 = CachedImageReader(img2)

        lafs1, kpts1, scores1, descs1 = self.extractor(
            path1, image_reader=image_reader1
        )
        descs1 = descs1.T

        rotation_results: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for angle in self.conf.angles:
            if angle == 0:
                rotation = None
            else:
                rotation = RotationConfig(angles=[angle])
            lafs2, kpts2, scores2, descs2 = self.extractor(
                path2, rotation=rotation, image_reader=image_reader2
            )
            descs2 = descs2.T  # Shape(N, dim) -> Shape(dim, N)

            data = {
                "keypoints0": torch.from_numpy(kpts1)
                .float()
                .unsqueeze(0)
                .to(self.device),
                "keypoints1": torch.from_numpy(kpts2)
                .float()
                .unsqueeze(0)
                .to(self.device),
                "descriptors0": torch.from_numpy(descs1)
                .float()
                .unsqueeze(0)
                .to(self.device),
                "descriptors1": torch.from_numpy(descs2)
                .float()
                .unsqueeze(0)
                .to(self.device),
                "scores0": torch.from_numpy(scores1)
                .float()
                .unsqueeze(0)
                .to(self.device),
                "scores1": torch.from_numpy(scores2)
                .float()
                .unsqueeze(0)
                .to(self.device),
                "shape0": (1, 1, *shape1),
                "shape1": (1, 1, *shape2),
            }

            preds = self.model(data)
            preds = {k: v[0].detach().cpu().numpy() for k, v in preds.items()}

            matches = preds["matches0"]
            scores = preds["matching_scores0"]

            valid = matches > -1

            idxs = np.concatenate(
                [np.where(valid)[0][None].T, matches[valid][None].T], axis=1
            )  # Shape(#matches, 2)

            mkpts1 = kpts1[idxs[:, 0]].copy()
            mkpts2 = kpts2[idxs[:, 1]].copy()
            scores = scores[valid].copy()
            rotation_results.append((mkpts1, mkpts2, scores))

        if self.conf.output_type == "concat":
            mkpts1 = np.concatenate(
                [_mkpts1 for _mkpts1, _, _ in rotation_results], axis=0
            )
            mkpts2 = np.concatenate(
                [_mkpts2 for _, _mkpts2, _ in rotation_results], axis=0
            )
            scores = np.concatenate(
                [_scores for _, _, _scores in rotation_results], axis=0
            )
        else:
            matching_counts = np.array(
                [len(_mkpts1) for _mkpts1, _, _ in rotation_results]
            )
            best = int(matching_counts.argsort()[-1])
            mkpts1, mkpts2, scores = rotation_results[best]

        if self.conf.min_matches is None or len(mkpts1) >= self.conf.min_matches:
            matched_keypoint_storage.add(path1, path2, mkpts1, mkpts2, scores=scores)

    def match(
        self,
        descs1: np.ndarray,
        descs2: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError
