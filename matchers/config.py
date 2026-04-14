from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel

from core import ConfigBlock
from features.config import LocalFeatureConfig, XFeatConfig
from models.config import HuggingFaceModelConfig, MASt3RModelConfig
from postprocesses.config import (
    MatchingFilterConfig,
    NMSConfig,
    PANetRefinerConfig,
    RANSACConfig,
)
from preprocesses.config import PairedPreResizeConfig, ResizeConfig


class DualSoftmaxMatcherConfig(BaseModel):
    normalize: bool = False
    inv_temp: float = 1.0
    threshold: float = 0.0

    min_matches: Optional[int] = None
    use_overlap_region_cropper: bool = False

    def __str__(self) -> str:
        return f"ds(th={self.threshold}, norm={self.normalize}, inv_temp={self.inv_temp}, min={self.min_matches}, oe={self.use_overlap_region_cropper})"


class MagicLeapSuperGlueConfig(BaseModel):
    weight_path: str
    min_matches: Optional[int] = None
    use_overlap_region_cropper: bool = False

    weights: str = "outdoor"
    sinkhorn_iterations: int = 30
    match_threshold: float = 0.2

    def __str__(self) -> str:
        return f"magicleap_superglue(iter={self.sinkhorn_iterations}, th={self.match_threshold}, min={self.min_matches}, oe={self.use_overlap_region_cropper})"


class NNMatcherConfig(BaseModel):
    match_mode: str = "smnn"
    min_matches: Optional[int] = None
    use_overlap_region_cropper: bool = False

    th: Optional[float] = None

    def __str__(self) -> str:
        return f"nn(match_mode={self.match_mode}, th={self.th}, min={self.min_matches}, oe={self.use_overlap_region_cropper})"


class NNDoubleSoftmaxConfig(BaseModel):
    match_mode: str = "smnn"
    min_matches: Optional[int] = None
    use_overlap_region_cropper: bool = False

    temperature: float = 1.0
    th: Optional[float] = None

    def __str__(self) -> str:
        return f"nnds(match_mode={self.match_mode}, th={self.th}, min={self.min_matches}, temp={self.temperature}, oe={self.use_overlap_region_cropper})"


class OpenGlueConfig(BaseModel):
    experiment_path: str
    checkpoint_name: str
    descriptor_dim: int
    min_matches: Optional[int] = None
    use_overlap_region_cropper: bool = False

    max_features: int = 2048
    resize_to: str = "original"

    match_threshold: Optional[float] = None  # default: 0.2
    topk: Optional[int] = None  # TODO: support

    def __str__(self) -> str:
        return f"openglue(num={self.max_features}, th={self.match_threshold}, min={self.min_matches}, oe={self.use_overlap_region_cropper})"


class LightGlueConfig(BaseModel):
    weight_path: str
    feature_name: Literal[
        "aliked",
        "dedodeb",
        "dedodeg",
        "disk",
        "dog_affnet_hardnet",
        "doghardnet",
        "keynet_affnet_hardnet",
        "sift",
        "superpoint",
    ]

    min_matches: Optional[int] = None
    use_overlap_region_cropper: bool = False

    width_confidence: float = -1.0
    depth_confidence: float = -1.0
    mp: bool = False

    def __str__(self) -> str:
        return f"lightglue(feature={self.feature_name}, min={self.min_matches}, oe={self.use_overlap_region_cropper})"

    def get_params(self) -> dict:
        return {
            "width_confidence": self.width_confidence,
            "depth_confidence": self.depth_confidence,
            "mp": self.mp,
        }


class GIMLightGlueConfig(BaseModel):
    weight_path: str

    min_matches: Optional[int] = None
    use_overlap_region_cropper: bool = False

    def __str__(self) -> str:
        return f"gim_lightglue(min={self.min_matches}, oe={self.use_overlap_region_cropper})"


class SteererConfig(BaseModel):
    weight_path: str
    matcher_type: Literal[
        "MaxMatchesMatcher",
        "MaxSimilarityMatcher",
        "SubsetMatcher",
        "ContinuousMaxMatchesMatcher",
        "ContinuousMatxSimilarityMatcher",
        "ContinuousSubsetMatcher",
    ]
    steerer_order: Optional[int] = None
    angles: Optional[list[float]] = None
    so2: bool = False

    normalize: bool = True
    inv_temp: float = 20.0
    threshold: float = 0.01

    min_matches: Optional[int] = None
    use_overlap_region_cropper: bool = False

    def __str__(self) -> str:
        return f"steerer(weight={self.weight_path}, type={self.matcher_type}, order={self.steerer_order}, angles={self.angles}, min={self.min_matches}, oe={self.use_overlap_region_cropper})"


class LocalFeatureMatcherConfig(BaseModel):
    # magicleap_superglue | magicleap_superglue_rotation | nn |
    # nn_double_softmax | openglue | lightglue | dual_softmax | steerer
    type: str = "nn"

    panet: Optional[PANetRefinerConfig] = None

    magicleap_superglue: Optional[MagicLeapSuperGlueConfig] = None
    nn: Optional[NNMatcherConfig] = None
    nn_double_softmax: Optional[NNDoubleSoftmaxConfig] = None
    openglue: Optional[OpenGlueConfig] = None
    lightglue: Optional[LightGlueConfig] = None
    gim_lightglue: Optional[GIMLightGlueConfig] = None
    dual_softmax: Optional[DualSoftmaxMatcherConfig] = None
    steerer: Optional[SteererConfig] = None


class AdaMatcherConfig(BaseModel):
    weight_path: str
    min_matches: Optional[int] = None

    thr: Optional[float] = None

    confidence_threshold: Optional[float] = None
    topk: Optional[int] = None

    resize: Optional[ResizeConfig] = None

    def __str__(self) -> str:
        return f"adamatcher(th={self.thr}, conf_th={self.confidence_threshold}, topk={self.topk}, min={self.min_matches}, resize={str(self.resize)})"


class ASpanFormerConfig(BaseModel):
    weight_path: str
    min_matches: Optional[int] = None

    thr: Optional[float] = None
    resize: Optional[ResizeConfig] = None

    def __str__(self) -> str:
        return f"aspanformer(th={self.thr}, min={self.min_matches}, resize={str(self.resize)})"


class DKMConfig(BaseModel):
    weight_path: str
    model_type: str = "DKMv3_outdoor"
    min_matches: Optional[int] = None

    height: Optional[int] = None
    width: Optional[int] = None
    upsample_preds: bool = True
    sample_mode: str = "threshold_balanced"

    upsample_height: Optional[int] = None
    upsample_width: Optional[int] = None

    sample_threshold: Optional[float] = None
    sample_nums: int = 10000

    confidence_threshold: Optional[float] = None
    topk: Optional[int] = None

    nms: Optional[NMSConfig] = None
    panet: Optional[PANetRefinerConfig] = None
    verification: Optional[RANSACConfig] = None

    def __str__(self) -> str:
        return f"dkm(num={self.sample_nums}, conf={self.confidence_threshold}, topk={self.topk}, th={self.sample_threshold}, height={self.height}, width={self.width}, min={self.min_matches}, verification={str(self.verification)})"


class DKMRotationConfig(BaseModel):
    dkm: DKMConfig
    angles: list[int]
    pre_resize: Optional[ResizeConfig] = None

    min_matches: Optional[int] = None

    nms: Optional[NMSConfig] = None

    def __str__(self) -> str:
        return f"dkm_rot(angles={self.angles}, dkm={str(self.dkm)})"


class ECOTRConfig(BaseModel):
    weight_path: str
    min_matches: Optional[int] = None

    max_kpts_num: Optional[int] = None
    aspect_ratios: Optional[list[float]] = None
    cycle: bool = False
    level: str = "fine"

    uncertainty_threshold: float = 1e-2  # matches[matches[:, -1] < th]

    resize: Optional[ResizeConfig] = None

    def __str__(self) -> str:
        return f"ecotr(num={self.max_kpts_num}, th={self.uncertainty_threshold}, min={self.min_matches}, resize={str(self.resize)})"


class EfficientLoFTRConfig(BaseModel):
    weight_path: str
    config_file_path: Optional[str] = None
    min_matches: Optional[int] = None

    match_coarse_thr: Optional[float] = None

    confidence_threshold: Optional[float] = None
    topk: Optional[int] = None

    resize: Optional[ResizeConfig] = None
    paired_pre_resize: Optional[PairedPreResizeConfig] = None

    def __str__(self) -> str:
        return f"efficientloftr(min={self.min_matches}, th={self.confidence_threshold}, topk={self.topk}, resize={str(self.resize)})"


class GIMDKMConfig(BaseModel):
    weight_path: str
    model_type: str = "DKMv3_outdoor"
    min_matches: Optional[int] = None

    height: int = 672
    width: int = 896
    upsample_preds: bool = True
    sample_mode: str = "threshold_balanced"

    upsample_height: Optional[int] = None
    upsample_width: Optional[int] = None

    sample_threshold: Optional[float] = None
    sample_nums: int = 5000

    confidence_threshold: Optional[float] = None
    topk: Optional[int] = None

    nms: Optional[NMSConfig] = None
    panet: Optional[PANetRefinerConfig] = None
    verification: Optional[RANSACConfig] = None

    def __str__(self) -> str:
        return f"gim_dkm(num={self.sample_nums}, conf={self.confidence_threshold}, topk={self.topk}, th={self.sample_threshold}, height={self.height}, width={self.width}, min={self.min_matches}, verification={str(self.verification)})"


class GIMLoFTRConfig(BaseModel):
    weight_path: str
    min_matches: Optional[int] = None

    confidence_threshold: Optional[float] = None
    topk: Optional[int] = None

    resize: Optional[ResizeConfig] = None
    paired_pre_resize: Optional[PairedPreResizeConfig] = None

    def __str__(self) -> str:
        return f"gim_loftr(weight={self.weight_path}, min={self.min_matches}, th={self.confidence_threshold}, topk={self.topk}, resize={str(self.resize)})"


class LoFTRConfig(BaseModel):
    weight_path: str
    min_matches: Optional[int] = None

    confidence_threshold: Optional[float] = None
    topk: Optional[int] = None

    resize: Optional[ResizeConfig] = None
    paired_pre_resize: Optional[PairedPreResizeConfig] = None

    def __str__(self) -> str:
        return f"loftr(weight={self.weight_path}, min={self.min_matches}, th={self.confidence_threshold}, topk={self.topk}, resize={str(self.resize)})"


class MagicLeapSuperGlueRotationConfig(BaseModel):
    ml_superglue: MagicLeapSuperGlueConfig
    local_feature: LocalFeatureConfig
    angles: list[int]

    output_type: str = "concat"  # concat | best_angle
    min_matches: Optional[int] = None

    def __str__(self) -> str:
        return f"magicleap_superglue_rot(angles={self.angles}, superglue={str(self.ml_superglue)})"


class MatchformerConfig(BaseModel):
    weight_path: str
    config_file_path: Optional[str] = None
    min_matches: Optional[int] = None

    backbone_type: str = "largela"
    scens: str = "outdoor"

    use_pad_mask: bool = False
    match_coarse_thr: Optional[float] = None

    resize: Optional[ResizeConfig] = None

    confidence_threshold: Optional[float] = None
    topk: Optional[int] = None

    def __str__(self) -> str:
        return f"matchformer(thr={self.match_coarse_thr}, conf_th={self.confidence_threshold}, topk={self.topk}, min={self.min_matches}, resize={str(self.resize)})"


class OmniGlueONNXConfig(BaseModel):
    weight_path: str
    weight_path_superpoint: str
    weight_path_dinov2: str
    min_matches: Optional[int] = None

    max_keypoints: int = 2048

    resize: Optional[ResizeConfig] = None

    confidence_threshold: Optional[float] = None
    topk: Optional[int] = None

    def __str__(self) -> str:
        return f"omniglue(thr={self.confidence_threshold}, topk={self.topk})"


class QuadTreeConfig(BaseModel):
    weight_path: str
    min_matches: Optional[int] = None

    confidence_threshold: Optional[float] = None
    topk: Optional[int] = None

    resize: Optional[ResizeConfig] = None
    paired_pre_resize: Optional[PairedPreResizeConfig] = None

    def __str__(self) -> str:
        return f"quadtree(thr={self.confidence_threshold}, topk={self.topk}, min={self.min_matches}, resize={str(self.resize)})"


class RoMaConfig(BaseModel):
    weight_path: str
    dinov2_weight_path: str

    model_type: str = "outdoor"
    min_matches: Optional[int] = None

    coarse_res: int = 560
    upsample_res: int = 864

    # height: Optional[int] = None
    # width: Optional[int] = None
    upsample_preds: bool = True
    # sample_mode: str = 'threshold'

    sample_threshold: Optional[float] = None
    sample_nums: int = 10000

    confidence_threshold: Optional[float] = None
    topk: Optional[int] = None

    nms: Optional[NMSConfig] = None

    def __str__(self) -> str:
        return f"roma(num={self.sample_nums}, th={self.sample_threshold}, coarse_res={self.coarse_res}, upsample_res={self.upsample_res}, min={self.min_matches})"


class SE2LoFTRConfig(BaseModel):
    weight_path: str
    config_file_path: Optional[str] = None
    min_matches: Optional[int] = None

    match_coarse_thr: Optional[float] = None

    confidence_threshold: Optional[float] = None
    topk: Optional[int] = None

    resize: Optional[ResizeConfig] = None
    paired_pre_resize: Optional[PairedPreResizeConfig] = None

    verification: Optional[RANSACConfig] = None

    def __str__(self) -> str:
        return f"se2loftr(min={self.min_matches}, th={self.confidence_threshold}, topk={self.topk}, resize={str(self.resize)}, verification={str(self.verification)})"


class XFeatStarMatcherConfig(BaseModel):
    xfeat: XFeatConfig

    min_matches: Optional[int] = None
    use_overlap_region_cropper: bool = False

    resize: Optional[ResizeConfig] = None

    def __str__(self) -> str:
        return f"xfeatstar(min={self.min_matches}, xfeat={str(self.xfeat)}, resize={str(self.resize)})"


class MASt3RMatcherConfig(BaseModel):
    mast3r: MASt3RModelConfig
    min_matches: Optional[int] = None
    size: int = 512
    subsample: int = 8
    pixel_tol: int = 5
    match_threshold: float = 1.001
    match_topk: Optional[int] = None

    def __str__(self) -> str:
        return f"mast3r(min={self.min_matches}, size={self.size})"


class MASt3RC2FMatcherConfig(BaseModel):
    mast3r: MASt3RModelConfig
    min_matches: Optional[int] = None
    size: int = 512
    subsample: int = 8
    pixel_tol: int = 5
    match_threshold: float = 1.001
    match_topk: Optional[int] = None

    max_image_size: int = 1024
    max_batch_size: int = 8

    def __str__(self) -> str:
        return f"mast3r_c2f(min={self.min_matches}, size={self.size})"


class DetectorFreeMatcherConfig(ConfigBlock):
    # adamatcher | aspanformer | dkm | dkm_rotation | ecotr | efficientloftr | loftr |
    # magicleap_superglue_rotation | matchformer | quadtree | se2loftr | xfeat_star | mast3r | mast3r_c2f | gim_loftr
    type: str = "loftr"

    apply_round: bool = True
    mkpts_decoupling_method: Literal["imc2023", "detector_free_sfm"] = "imc2023"
    cropper_type: Literal["overlap", "roi", "overlap-or-roi", "ignore"] = "overlap"
    matching_filter: Optional[MatchingFilterConfig] = None

    adamatcher: Optional[AdaMatcherConfig] = None
    aspanformer: Optional[ASpanFormerConfig] = None
    dkm: Optional[DKMConfig] = None
    dkm_rotation: Optional[DKMRotationConfig] = None
    ecotr: Optional[ECOTRConfig] = None
    efficientloftr: Optional[EfficientLoFTRConfig] = None
    gim_dkm: Optional[GIMDKMConfig] = None
    gim_loftr: Optional[GIMLoFTRConfig] = None
    loftr: Optional[LoFTRConfig] = None
    magicleap_superglue_rotation: Optional[MagicLeapSuperGlueRotationConfig] = None
    mast3r: Optional[MASt3RMatcherConfig] = None
    mast3r_c2f: Optional[MASt3RC2FMatcherConfig] = None
    matchformer: Optional[MatchformerConfig] = None
    omniglue_onnx: Optional[OmniGlueONNXConfig] = None
    quadtree: Optional[QuadTreeConfig] = None
    roma: Optional[RoMaConfig] = None
    se2loftr: Optional[SE2LoFTRConfig] = None
    xfeat_star: Optional[XFeatStarMatcherConfig] = None


class PreMatcherConfig(BaseModel):
    type: Literal["local_feature", "detector_free"]
    local_feature: Optional[LocalFeatureConfig] = None
    local_feature_matcher: Optional[LocalFeatureMatcherConfig] = None
    detector_free_matcher: Optional[DetectorFreeMatcherConfig] = None

    def __str__(self) -> str:
        if self.type == "local_feature":
            assert self.local_feature
            assert self.local_feature_matcher
            return f"prematcher(feat={self.local_feature.type}, matcher={self.local_feature_matcher.type})"
        assert self.type == "detector_free"
        assert self.detector_free_matcher
        return f"prematcher(matcher={self.detector_free_matcher.type})"


class LIMAPMatcherConfig(BaseModel):
    matcher_method: str
    extractor_method: str
    matcher_weight_path: Optional[str] = None
    extractor_weight_path: Optional[str] = None

    def __str__(self) -> str:
        return (
            f"limap(matcher={self.matcher_method}, extractor={self.extractor_method})"
        )


class Line2DFeatureMatcherConfig(BaseModel):
    type: Literal["limap"]

    limap: Optional[LIMAPMatcherConfig] = None

    def __str__(self) -> str:
        c = getattr(self, self.type)
        return f"line2d({self.type}={str(c)})"


class VGGTMatcherConfig(ConfigBlock):
    model: HuggingFaceModelConfig
    min_matches: Optional[int] = None

    load_fixed_weight: bool = False
    use_custom_vggt: bool = False

    target_size: int = 518
    conf_score_threshold: float = 0.2
    vis_score_threshold: float = 0.2
    filtering_method: Literal["conf", "vis", "conf&vis", "conf|vis"] = "conf&vis"
    track_iters: int | None = None

    def __str__(self) -> str:
        return f"vggt(min={self.min_matches})"


class MASt3RMPSFMSparseMatcherConfig(ConfigBlock):
    model: MASt3RModelConfig
    size: int = 512
    subsample: int = 8
    nn_score_threshold: float = 0.85

    min_matches: Optional[int] = None
    score_threshold: Optional[float] = 0.0


class MASt3RSparseMatcherConfig(ConfigBlock):
    model: MASt3RModelConfig
    size: int = 512
    min_matches: Optional[int] = None
    match_threshold: float = 1.001
    match_topk: Optional[int] = None


class MASt3RHybridMatcherConfig(ConfigBlock):
    model: MASt3RModelConfig
    size: int = 512

    prefer_sparse_matches: bool = False
    prefer_sparse_kernel_size: int = 5

    # For semi-dense matching
    dense_min_matches: Optional[int] = None
    dense_subsample: int = 8
    dense_pixel_tol: int = 5
    dense_match_threshold: float = 1.001
    dense_match_topk: Optional[int] = None

    # For sparse matching
    sparse_min_matches: Optional[int] = None
    sparse_match_threshold: float = 1.001
    sparse_match_topk: Optional[int] = None


class PointTrackingMatcherConfig(ConfigBlock):
    type: Literal[
        "vggt",
        "mast3r_mpsfm_sparse",
        "mast3r_sparse",
        "mast3r_hybrid",
    ]
    local_features: list[LocalFeatureConfig] = []

    # For backward compatibility, 'v1' is set by default, but 'v2' should be set
    impl_version: Literal["v1", "v2"] = "v1"

    apply_round: bool = True
    matching_filter: Optional[MatchingFilterConfig] = None

    vggt: Optional[VGGTMatcherConfig] = None
    mast3r_mpsfm_sparse: Optional[MASt3RMPSFMSparseMatcherConfig] = None
    mast3r_sparse: Optional[MASt3RSparseMatcherConfig] = None
    mast3r_hybrid: Optional[MASt3RHybridMatcherConfig] = None
