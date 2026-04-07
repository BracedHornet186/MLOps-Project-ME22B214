from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel

from core import ConfigBlock
from models.config import HuggingFaceModelConfig


class RANSACConfig(BaseModel):
    threshold: float = 0.1845
    confidence: float = 0.999999
    max_iters: int = 100000
    min_inliers: Optional[int] = None  # TODO

    method: Optional[str] = None

    def __str__(self) -> str:
        if self.method:
            return f"ransac(method={self.method}, th={self.threshold}, conf={self.confidence}, max_iters={self.max_iters})"
        return f"ransac(th={self.threshold}, conf={self.confidence}, max_iters={self.max_iters})"


class PoseLibRANSACConfig(BaseModel):
    max_iterations: int = 100000
    min_iterations: int = 1000
    dyn_num_trials_mult: float = 3.0
    success_prob: float = 0.9999
    max_reproj_error: float = 12.0
    max_epipolar_error: float = 1.0
    seed: int = 0
    progressive_sampling: bool = False
    max_prosac_iterations: int = 100000
    real_focal_check: bool = False

    min_inliers: Optional[int] = None

    def __str__(self) -> str:
        return f"poselib(max_epi_error={self.max_epipolar_error}, prob={self.success_prob}, max_iters={self.max_iterations})"

    def to_poselib_ransac_options(self) -> dict:
        d = self.model_dump().copy()
        del d["min_inliers"]
        return d


class FSNetConfig(BaseModel):
    weight_path: str
    config_path: str
    f_path: Optional[str] = None

    ransac_list: list[RANSACConfig] = []
    min_inliers: Optional[int] = None  # TODO


class SSCConfig(BaseModel):
    num_ret_points: int
    tolerance: float = 0.1


class NMSConfig(BaseModel):
    type: str  # nms_fast | ssc

    distance: Optional[float] = None
    topk: Optional[int] = None

    ssc: Optional[SSCConfig] = None


class NCMNetConfig(BaseModel):
    weight_path: str

    inlier_ratio: Optional[float] = 0.1  # 0.1: Top10%
    inlier_prob_threshold: Optional[float] = None
    topk: Optional[int] = None


class PANetRefinerConfig(BaseModel):
    weight_path: str
    max_edge: int
    max_sum_edges: int
    batch_size: int = 64


class MatchingFilterConfig(BaseModel):
    type: Literal["ncmnet", "feature_track"]

    ncmnet: Optional[NCMNetConfig] = None


class MINIMAVerifierConfig(ConfigBlock):
    matcher_type: Literal["splg", "loftr"]

    ransac: RANSACConfig
    inlier_threshold: int = 50

    depth_model_weight_path: str
    sp_weight_path: Optional[str] = None
    lg_weight_path: Optional[str] = None
    loftr_weight_path: Optional[str] = None


class VGGTTwoViewGeometryPrunerConfig(ConfigBlock):
    model: HuggingFaceModelConfig

    world_points_conf_threshold: float = 1.5
    score_threshold: float = 0.1
    max_pairs: int = 20
