import numpy as np
import cv2
import kornia
import torch
from typing import List, Optional, Tuple
from features.base import lafs_to_keypoints
from postprocesses.config import NMSConfig
from postprocesses.ssc import ssc
from postprocesses.fast import nms_fast


def nms_local_features(
    lafs: torch.Tensor,
    scores: torch.Tensor,
    descs: torch.Tensor,
    img: np.ndarray,
    conf: NMSConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width, *_ = img.shape
    if conf.type == 'nms_fast':
        assert conf.distance is not None
        kpts = lafs_to_keypoints(lafs).cpu().numpy()
        _kpts = np.zeros((len(kpts), 3))
        _kpts[:, 0] = np.clip(kpts[:, 0], 0, width - 1)
        _kpts[:, 1] = np.clip(kpts[:, 1], 0, height - 1)
        _kpts[:, 2] = scores.cpu().numpy()
        _kpts = _kpts.transpose()   # Shape(3, N)
        _, nms_idx = nms_fast(_kpts, height, width, conf.distance)

        if conf.topk is not None:
            # Assume idx sorted by scores
            nms_idx = nms_idx[:conf.topk]
        keeps = nms_idx.tolist()
    elif conf.type == 'ssc':
        assert conf.ssc
        assert img is not None
        _kpts = lafs_to_keypoints(lafs).cpu().numpy()
        _scores = scores.cpu().numpy()
        cv2_kpts = to_cv2_keypoints(_kpts, _scores)
        H, W, *_ = img.shape
        keeps = ssc(cv2_kpts,
                    conf.ssc.num_ret_points,
                    conf.ssc.tolerance, W, H,
                    return_indices=True)
    else:
        raise ValueError(conf.type)
        
    lafs = lafs[keeps]
    scores = scores[keeps]
    descs = descs[keeps]

    return lafs, scores, descs


def nms_keypoints(
    kpts: np.ndarray,
    scores: np.ndarray,
    conf: NMSConfig,
    img: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if conf.type == 'ssc':
        assert conf.ssc
        assert img is not None
        cv2_kpts = to_cv2_keypoints(kpts, scores)
        H, W, *_ = img.shape
        keeps = ssc(cv2_kpts,
                    conf.ssc.num_ret_points,
                    conf.ssc.tolerance, W, H,
                    return_indices=True)
        kpts = kpts[keeps]
        scores = scores[keeps]
        return kpts, scores
    else:
        raise ValueError


def nms_matched_keypoints(
    mkpts1: np.ndarray,
    mkpts2: np.ndarray,
    scores: np.ndarray,
    conf: NMSConfig,
    img: Optional[np.ndarray] = None,
    image_shape: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if conf.type == 'ssc':
        assert conf.ssc
        cv2_kpts = to_cv2_keypoints(mkpts1, scores)
        if image_shape:
            H, W = image_shape
        else:
            assert img is not None
            H, W, *_ = img.shape
        keeps = ssc(cv2_kpts,
                    conf.ssc.num_ret_points,
                    conf.ssc.tolerance, W, H,
                    return_indices=True)
        mkpts1 = mkpts1[keeps]
        mkpts2 = mkpts2[keeps]
        scores = scores[keeps]
        return mkpts1, mkpts2, scores
    else:
        raise ValueError



def sort_keypoints_by_score(
    kpts: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    orders = -scores.argsort()
    kpts = kpts[orders]
    scores = scores[orders]
    return kpts, scores


def sort_matched_keypoints_by_score(
    mkpts1: np.ndarray,
    mkpts2: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(mkpts1) == len(mkpts2)
    orders = -scores.argsort()
    mkpts1 = mkpts1[orders]
    mkpts2 = mkpts2[orders]
    scores = scores[orders]
    return mkpts1, mkpts2, scores


def to_cv2_keypoints(
    kpts: np.ndarray,
    scores: np.ndarray
) -> List[cv2.KeyPoint]:
    assert len(kpts) == len(scores)
    cv2_kpts = [
        cv2.KeyPoint(x=p[0], y=p[1], size=1, angle=0, response=score, octave=0, class_id=0)
        for p, score in zip(kpts, scores)
    ]
    return cv2_kpts
