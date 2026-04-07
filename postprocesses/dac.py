"""Adapted from https://github.com/javrtg/DAC/blob/main/experiments/matching/eval_hpatches.py
"""
import numpy as np
from typing import Tuple

def reproj_err_and_match_cov(
    p1: np.ndarray,
    p2: np.ndarray,
    S1_inv: np.ndarray,
    S2_inv: np.ndarray,
    H21: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute:
        1) Euclidean distance between matched keypoints.
        2) Covariance matrix of the match: S1_propagated + S2 (i.e. assuming independence)

    The kps in the reference image, p1, are reprojected as:
        [p1_proj, 1] = y / y[3];    y = H @ [p1, 1]
    Thereby the Jacobian needed to lineraly propagate S1 is:
        J = d(p1_proj) / d(p1) = ( d(p1_proj) / d(y) ) * ( d(y) / d(p1) )
          = J1 * J2

    Args:                                                       Shape
        - p1: keypoints in reference image.                     (2,n)
        - p2: keypoints in target image.                        (2,n)
        - S1_inv: autocorr. matrix (inverse cov.) of p1.        (n,2,2)
        - S2_inv: autocorr. matrix (inverse cov.) of p2.        (n,2,2)
        - H21: ground-truth homography matrix.                  (3,3)

    Return:
        - dist: reprojection errors.                            (n,)
        - S: covariance matrices of the matches.                (n,2,2)
    """
    # get input covariances:
    # S1 = fast2x2inv(S1_inv)
    # S2 = fast2x2inv(S2_inv)
    S1 = np.linalg.inv(S1_inv)
    S2 = np.linalg.inv(S2_inv)

    error2d = (p1_proj_hom := (
        y := H21[:, :2] @ p1 + H21[:, -1:])[:-1] / y[-1:]
    ) - p2
    # euclidean distance error
    dist = np.sqrt(np.einsum('ij, ij -> j', error2d, error2d))

    # J1 (1st term of the chain-rule)
    J1 = np.repeat([[[1., 0., 0.], [0., 1., 0.]]], len(S1_inv), axis=0)
    J1[:, :, -1] = -p1_proj_hom.T
    J1 /= y[2, :, None, None]
    # complete the chain-rule:
    J = J1 @ H21[:, :2]
    # covariance of the match with linear propagation of S1:
    S = (J @ S1 @ J.transpose(0, 2, 1)) + S2
    return dist, S