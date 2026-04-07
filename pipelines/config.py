from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel

from clusterings.config import ClusteringConfig
from core import ConfigBlock
from features.config import Line2DFeatureConfig, LocalFeatureConfig
from localizers.config import LocalizerConfig, PostLocalizerConfig
from matchers.config import (
    DetectorFreeMatcherConfig,
    Line2DFeatureMatcherConfig,
    LocalFeatureMatcherConfig,
    MASt3RC2FMatcherConfig,
    MASt3RHybridMatcherConfig,
    MASt3RMatcherConfig,
    PointTrackingMatcherConfig,
    PreMatcherConfig,
)
from models.config import DoppelGangersModelConfig
from postprocesses.config import (
    FSNetConfig,
    MatchingFilterConfig,
    PANetRefinerConfig,
    PoseLibRANSACConfig,
    RANSACConfig,
    VGGTTwoViewGeometryPrunerConfig,
)
from preprocesses.config import (
    DeblurringConfig,
    DepthEstimationConfig,
    MaskingConfig,
    OrientationNormalizationConfig,
    OverlapRegionEstimatorConfig,
    SegmentationConfig,
)
from shortlists.config import (
    EnsembleShortlistGeneratorConfig,
    IMC2024ShortlistGeneratorConfig,
    ShortlistGeneratorConfig,
    ShortlistUpdaterConfig,
)


class PreMatchingConfig(BaseModel):
    matchers: list[PreMatcherConfig] = []

    filter_by_segmentation_if_provided: bool = False

    def __str__(self) -> str:
        matchers = [str(m) for m in self.matchers]
        return f"prematching(matchers={matchers}, filter_by_seg={self.filter_by_segmentation_if_provided})"


class GeometricVerificationConfig(BaseModel):
    type: str = "colmap"  # colmap | custom | fsnet | poselib

    ransac: Optional[RANSACConfig] = None
    poselib_ransac: Optional[PoseLibRANSACConfig] = None
    fsnet: Optional[FSNetConfig] = None

    def __str__(self) -> str:
        return f"{self.type}(ransac={str(self.ransac)}, poselib={self.poselib_ransac})"


class TwoViewGeometryPruningConfig(BaseModel):
    type: Literal["doppelgangers", "vggt"]

    doppelgangers: Optional[DoppelGangersModelConfig] = None
    vggt: Optional[VGGTTwoViewGeometryPrunerConfig] = None

    def __str__(self) -> str:
        return f"pruning({str(getattr(self, self.type))})"


class ReconstructionConfig(BaseModel):
    camera_model: Literal["simple-radial", "simple-pinhole", "auto"] = (
        "simple-radial"  # simple-radial | simple-pinhole | auto
    )

    fill_zero_Rt: bool = False  # IMC2025: True->False
    fill_nan_Rt: bool = False
    fill_nearest_position: bool = False
    use_localize_sfm: bool = False
    use_localize_pixloc: bool = False

    mapper_min_model_size: Optional[int] = None  # 10 (default)
    mapper_max_num_models: Optional[int] = None  # 50 (default)
    mapper_min_num_matches: Optional[int] = None  # 15 (default)
    mapper_multiple_models: Optional[int] = None  # 0 | 1 (default)
    mapper_filter_max_reproj_error: Optional[int] = None  # 4 (default)

    set_scene_graph_center_node_to_init_image_id1: bool = False

    def __str__(self) -> str:
        return f"rec(cam={self.camera_model}, min_model_size={self.mapper_min_model_size}, max_num={self.mapper_max_num_models}, min_num_matches={self.mapper_min_num_matches}, fill_nan={self.fill_nan_Rt}, localize_sfm={self.use_localize_sfm}, filter_max_reproj_err={self.mapper_filter_max_reproj_error}, id1={self.set_scene_graph_center_node_to_init_image_id1})"

    def get_camera_model(self, unique_resolution_num: Optional[int] = None) -> str:
        if self.camera_model == "auto":
            if unique_resolution_num is None:
                print("[get_camera_model] auto -> simple-radial")
                return "simple-radial"
            elif unique_resolution_num == 1:
                print("[get_camera_model] auto -> simple-pinhole")
                return "simple-pinhole"
            else:
                print(
                    f"[get_camera_model] auto -> simple-radial (uniq-res={unique_resolution_num})"
                )
                return "simple-radial"
        return self.camera_model


class HLocMatchDenseConfig(BaseModel):
    max_error: int = 1
    cell_size: int = 1
    max_keypoints: Optional[int] = None

    use_local_features: bool = False

    def __str__(self) -> str:
        return f"hloc(max_error={self.max_error}, cell={self.cell_size}, max_kpts={self.max_keypoints}, use_lf={self.use_local_features})"


class DetectorBasedPipelineConfig(BaseModel):
    local_features: list[LocalFeatureConfig] = []
    shortlist_generator: Union[
        ShortlistGeneratorConfig, IMC2024ShortlistGeneratorConfig
    ] = ShortlistGeneratorConfig()
    matcher: LocalFeatureMatcherConfig = LocalFeatureMatcherConfig()

    filtering: Optional[MatchingFilterConfig] = None
    verification: GeometricVerificationConfig = GeometricVerificationConfig()
    reconstruction: ReconstructionConfig = ReconstructionConfig()

    seed: int = 1234


class DetectorFreePipelineConfig(BaseModel):
    shortlist_generator: ShortlistGeneratorConfig = ShortlistGeneratorConfig()
    matcher: DetectorFreeMatcherConfig = DetectorFreeMatcherConfig()

    hloc_match_dense: Optional[HLocMatchDenseConfig] = None
    filtering: Optional[MatchingFilterConfig] = None
    verification: GeometricVerificationConfig = GeometricVerificationConfig()
    reconstruction: ReconstructionConfig = ReconstructionConfig()

    seed: int = 1234


class PreMatchingEnsemblePipelineConfig(BaseModel):
    shortlist_generator: ShortlistGeneratorConfig = ShortlistGeneratorConfig()

    pre_matcher: DetectorFreeMatcherConfig = DetectorFreeMatcherConfig()
    overlap_region_estimation: Optional[OverlapRegionEstimatorConfig] = None
    shortlist_updater: Optional[ShortlistUpdaterConfig] = None

    local_features: list[LocalFeatureConfig] = []
    local_feature_matchers: list[LocalFeatureMatcherConfig] = []

    detector_free_matchers: list[DetectorFreeMatcherConfig] = []

    hloc_match_dense: Optional[HLocMatchDenseConfig] = None
    filtering: Optional[MatchingFilterConfig] = None
    refinement: Optional[PANetRefinerConfig] = None
    verification: GeometricVerificationConfig = GeometricVerificationConfig()
    reconstruction: ReconstructionConfig = ReconstructionConfig()

    seed: int = 1234


class SimpleEnsemblePipelineConfig(BaseModel):
    shortlist_generator: ShortlistGeneratorConfig = ShortlistGeneratorConfig()

    local_features: list[LocalFeatureConfig] = []
    local_feature_matchers: list[LocalFeatureMatcherConfig] = []

    detector_free_matchers: list[DetectorFreeMatcherConfig] = []

    hloc_match_dense: Optional[HLocMatchDenseConfig] = None
    filtering: Optional[MatchingFilterConfig] = None
    verification: GeometricVerificationConfig = GeometricVerificationConfig()
    reconstruction: ReconstructionConfig = ReconstructionConfig()

    seed: int = 1234


class ANNEnsemblePipelineConfig(BaseModel):
    local_features: list[LocalFeatureConfig] = []
    local_feature_matchers: list[LocalFeatureMatcherConfig] = []

    detector_free_matchers: list[DetectorFreeMatcherConfig] = []

    filtering: Optional[MatchingFilterConfig] = None
    verification: GeometricVerificationConfig = GeometricVerificationConfig()
    reconstruction: ReconstructionConfig = ReconstructionConfig()

    seed: int = 1234


class IMC2024PipelineConfig(BaseModel):
    shortlist_generator: ShortlistGeneratorConfig = ShortlistGeneratorConfig()

    deblurring: Optional[DeblurringConfig] = None
    orientation_normalization: Optional[OrientationNormalizationConfig] = None
    segmentation: Optional[SegmentationConfig] = None
    depth_estimation: Optional[DepthEstimationConfig] = None
    pre_matching: Optional[PreMatchingConfig] = None
    overlap_region_estimation: Optional[OverlapRegionEstimatorConfig] = None
    masking: Optional[MaskingConfig] = None
    shortlist_updater: Optional[ShortlistUpdaterConfig] = None

    local_features: list[LocalFeatureConfig] = []
    local_feature_matchers: list[LocalFeatureMatcherConfig] = []

    detector_free_matchers: list[DetectorFreeMatcherConfig] = []

    line2d_features: list[Line2DFeatureConfig] = []
    line2d_matchers: list[Line2DFeatureMatcherConfig] = []

    post_localizer: Optional[PostLocalizerConfig] = None

    hloc_match_dense: Optional[HLocMatchDenseConfig] = None
    filtering: Optional[MatchingFilterConfig] = None
    refinement: Optional[PANetRefinerConfig] = None
    verification: GeometricVerificationConfig = GeometricVerificationConfig()
    pruning: Optional[TwoViewGeometryPruningConfig] = None
    reconstruction: ReconstructionConfig = ReconstructionConfig()

    seed: int = 1234


class DUSt3RPipelineConfig(BaseModel):
    shortlist_generator: ShortlistGeneratorConfig = ShortlistGeneratorConfig()

    pre_matching: Optional[PreMatchingConfig] = None
    overlap_region_estimation: Optional[OverlapRegionEstimatorConfig] = None
    masking: Optional[MaskingConfig] = None
    shortlist_updater: Optional[ShortlistUpdaterConfig] = None

    line2d_features: list[Line2DFeatureConfig] = []

    reconstruction: ReconstructionConfig = ReconstructionConfig()

    seed: int = 1234


class LocalizerPipelineConfig(BaseModel):
    localizer: LocalizerConfig
    depth_estimation: Optional[DepthEstimationConfig] = None
    shortlist_generator: ShortlistGeneratorConfig = ShortlistGeneratorConfig()

    seed: int = 1234


class IMC2025PipelineConfig(BaseModel):
    shortlist_generator: ShortlistGeneratorConfig = ShortlistGeneratorConfig()

    deblurring: Optional[DeblurringConfig] = None
    orientation_normalization: Optional[OrientationNormalizationConfig] = None
    segmentation: Optional[SegmentationConfig] = None
    depth_estimation: Optional[DepthEstimationConfig] = None
    pre_matching: Optional[PreMatchingConfig] = None
    overlap_region_estimation: Optional[OverlapRegionEstimatorConfig] = None
    masking: Optional[MaskingConfig] = None
    shortlist_updater: Optional[ShortlistUpdaterConfig] = None

    local_features: list[LocalFeatureConfig] = []
    local_feature_matchers: list[LocalFeatureMatcherConfig] = []

    detector_free_matchers: list[DetectorFreeMatcherConfig] = []

    point_tracking_matchers: list[PointTrackingMatcherConfig] = []

    line2d_features: list[Line2DFeatureConfig] = []
    line2d_matchers: list[Line2DFeatureMatcherConfig] = []

    post_localizer: Optional[PostLocalizerConfig] = None

    hloc_match_dense: Optional[HLocMatchDenseConfig] = None
    filtering: Optional[MatchingFilterConfig] = None
    refinement: Optional[PANetRefinerConfig] = None
    verification: GeometricVerificationConfig = GeometricVerificationConfig()
    pruning: Optional[TwoViewGeometryPruningConfig] = None
    reconstruction: ReconstructionConfig = ReconstructionConfig()

    clustering: Optional[ClusteringConfig] = None

    use_glomap: bool = False
    seed: int = 1234


class IMC2025MASt3RPipelineConfig(ConfigBlock):
    clustering: ClusteringConfig
    shortlist_generator_in_clustering: Optional[ShortlistGeneratorConfig] = None

    shortlist_generator: ShortlistGeneratorConfig
    matcher: MASt3RMatcherConfig
    verification: GeometricVerificationConfig
    reconstruction: ReconstructionConfig

    matcher_c2f: Optional[MASt3RC2FMatcherConfig] = None
    matcher_hybrid: Optional[PointTrackingMatcherConfig] = None
    overlap_region_estimation: Optional[OverlapRegionEstimatorConfig] = None
    cropper_type: Literal["overlap", "ignore"] = "ignore"

    clustering_neighbor_metric: Literal["dist", "pair"] = "dist"

    matching_stage_mode: Literal[
        "complementary",
        "c2f_override",
        "hybrid_matcher_override",
    ] = "complementary"

    round_matched_keypoints: bool = True

    use_glomap: bool = False
    seed: int = 2025
