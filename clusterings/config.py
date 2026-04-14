from __future__ import annotations

from typing import Literal, Optional, TypeAlias

from core import ConfigBlock
from matchers.config import MASt3RMatcherConfig
from models.config import (
    APGeMConfig,
    CVNetConfig,
    DINOv2SALADConfig,
    HuggingFaceModelConfig,
    ISCModelConfig,
    MASt3RRetrievalModelConfig,
    MoGeModelConfig,
    PatchNetVLADModelConfig,
)
from retrievers.config import RetrieverConfig

CluasteringType: TypeAlias = Literal[
    "connected_component",
    "dbscan",
    "debug_array_split",
    "vggt",
    "vggt_fps",
    "mast3r_fps",
]


class GlobalDescriptorExtractorConfig(ConfigBlock):
    global_desc_model: str | None

    batch_size: int = 16

    apgem: Optional[APGeMConfig] = None
    cvnet: Optional[CVNetConfig] = None
    dinov2: Optional[HuggingFaceModelConfig] = None
    dinov2_salad: Optional[DINOv2SALADConfig] = None
    patchnetvlad: Optional[PatchNetVLADModelConfig] = None
    mast3r_spoc: Optional[MASt3RRetrievalModelConfig] = None
    mast3r_retrieval_spoc: Optional[MASt3RRetrievalModelConfig] = None
    mast3r_retrieval_asmk: Optional[MASt3RRetrievalModelConfig] = None
    moge: Optional[MoGeModelConfig] = None
    siglip2: Optional[HuggingFaceModelConfig] = None
    isc: Optional[ISCModelConfig] = None


class DBSCANClusteringConfig(ConfigBlock):
    global_desc: GlobalDescriptorExtractorConfig
    eps: float = 0.5
    min_samples: int = 5


class ConnectedComponentClusteringConfig(ConfigBlock):
    global_desc: Optional[GlobalDescriptorExtractorConfig] = None
    retriever: Optional[RetrieverConfig] = None

    topk: int = 10
    dist_threshold: float = 0.2
    degree_threshold: Optional[int] = None

    min_cluster_size: int = 3
    use_noisy_cluster_as_one_cluster: bool = False


class DebugArraySplitClusteringConfig(ConfigBlock):
    n_clusters: int = 3


class VGGTClusteringConfig(ConfigBlock):
    model: HuggingFaceModelConfig
    image_size: int = 518  # 14n
    window_size: int = 6

    num_cycles: int = 5

    depth_score_threshold: Optional[float] = None
    world_points_score_threshold: Optional[float] = None

    seed: int = 2025
    verbose: bool = False


class VGGTFPSClusteringConfig(ConfigBlock):
    global_desc: GlobalDescriptorExtractorConfig

    vggt: HuggingFaceModelConfig
    image_size: int = 518  # 14n
    window_size: int = 6
    depth_score_threshold: Optional[float] = None
    world_points_score_threshold: Optional[float] = None

    mast3r_matcher: Optional[MASt3RMatcherConfig] = None

    fps_n: int = 10
    fps_dist_threshold: float = 0.9

    min_cluster_size: int = 3
    use_noisy_cluster_as_one_cluster: bool = False

    seed: int = 2025
    verbose: bool = False


class MASt3RFPSClusteringConfig(ConfigBlock):
    mast3r_retrieval_model: MASt3RRetrievalModelConfig
    mast3r_matcher: MASt3RMatcherConfig

    fps_n: int = 10
    fps_dist_threshold: float = 0.9
    initial_point_type: Literal["dbscan-bug", "dbscan", "simsum_max"] = "dbscan-bug"

    limit_depth_for_exhaustive_matching: int = 1  # NOTE: depth starts with 1
    topk_for_partial_matching: int = 10
    num_query_from_target: int = 5

    min_cluster_size: int = 3
    use_noisy_cluster_as_one_cluster: bool = False

    seed: int = 2025
    verbose: bool = False


class ClusteringConfig(ConfigBlock):
    type: CluasteringType

    connected_component: Optional[ConnectedComponentClusteringConfig] = None
    dbscan: Optional[DBSCANClusteringConfig] = None
    debug_array_split: Optional[DebugArraySplitClusteringConfig] = None
    vggt: Optional[VGGTClusteringConfig] = None
    vggt_fps: Optional[VGGTFPSClusteringConfig] = None
    mast3r_fps: Optional[MASt3RFPSClusteringConfig] = None
