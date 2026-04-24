from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel

from scripts.core import ConfigBlock
from models.config import (
    APGeMConfig,
    CVNetConfig,
    DINOv2SALADConfig,
    DoppelGangersModelConfig,
    HuggingFaceModelConfig,
    ISCModelConfig,
    MASt3RRetrievalModelConfig,
    MoGeModelConfig,
    PatchNetVLADModelConfig,
)
from postprocesses.config import RANSACConfig
from retrievers.config import MAST3RRetrievalASMKRetrieverConfig

FilePath = Union[str, Path]


class EnsembleShortlistGeneratorConfig(ConfigBlock):
    shortlist_generators: list[ShortlistGeneratorConfig]
    all_pairs_fallback_threshold: int = 30


class ShortlistGeneratorConfig(ConfigBlock):
    # all_pairs | global_desc | cvnet_rerank | gdcc | random_walk |
    # doppelgangers | patchnetvlad | mast3r_retrieval_asmk | ensemble | isc
    type: str = "all_pairs"

    global_desc_model: Optional[str] = None  # apgem | cvnet
    global_desc_batch_size: int = 16
    global_desc_num_workers: int = 2
    global_desc_similar_distance_threshold: float = 0.6
    global_desc_topk: int = 20
    global_desc_fallback_threshold: int = 20
    global_desc_remove_swapped_pairs: bool = False  # Should be True
    global_desc_num_refills_when_no_matches: int = 0
    global_desc_compute_feats_if_fallback: bool = False

    global_desc_fps_n: Optional[int] = None  # 10
    global_desc_fps_k: Optional[int] = None
    global_desc_fps_dist_threshold: Optional[float] = None  # 0.9

    cvnet_rerank_alpha: float = 0.5
    cvnet_rerank_threshold: float = 1.0
    cvnet_rerank_batch_size: int = 16
    cvnet_rerank_num_workers: int = 2
    cvnet_rerank_image_size_height: Optional[int] = None
    cvnet_rerank_image_size_width: Optional[int] = None

    gdcc_degree_threshold: Optional[int] = None

    random_walk_matching_threshold: Optional[int] = None  # 30
    random_walk_num_trials_per_sample: Optional[int] = None  # 30
    random_walk_num_references_per_sample: Optional[int] = None  # 10
    random_walk_ransac: Optional[RANSACConfig] = None

    mast3r_retrieval_asmk_fallback_threshold: Optional[int] = None
    mast3r_retrieval_asmk_remove_swapped_pairs: bool = False  # Should be True
    mast3r_retrieval_asmk_make_pairs_fps_n: int = 0
    mast3r_retrieval_asmk_make_pairs_fps_k: int = 1
    mast3r_retrieval_asmk_make_pairs_fps_dist_threshold: Optional[float] = None

    apgem: Optional[APGeMConfig] = None
    cvnet: Optional[CVNetConfig] = None
    dinov2: Optional[HuggingFaceModelConfig] = None
    dinov2_salad: Optional[DINOv2SALADConfig] = None
    patchnetvlad: Optional[PatchNetVLADModelConfig] = None
    mast3r_spoc: Optional[MASt3RRetrievalModelConfig] = None
    mast3r_retrieval_spoc: Optional[MASt3RRetrievalModelConfig] = None
    mast3r_retrieval_asmk: Optional[MAST3RRetrievalASMKRetrieverConfig] = None
    moge: Optional[MoGeModelConfig] = None
    moge_depth_hog: Optional[MoGeModelConfig] = None
    moge_dinov2: Optional[MoGeModelConfig] = None
    doppelgangers: Optional[DoppelGangersModelConfig] = None
    siglip2: Optional[HuggingFaceModelConfig] = None
    isc: Optional[ISCModelConfig] = None

    ensemble: Optional[EnsembleShortlistGeneratorConfig] = None

    compat_version: Literal["v1"] = "v1"

    @classmethod
    def load_config(cls, path: FilePath) -> ShortlistGeneratorConfig:
        with open(path) as fp:
            return ShortlistGeneratorConfig.parse_obj(yaml.safe_load(fp))

    def __str__(self) -> str:
        kwargs = {}
        if self.global_desc_model:
            kwargs["model"] = self.global_desc_model
        kwargs["th"] = self.global_desc_similar_distance_threshold
        kwargs["topk"] = self.global_desc_topk
        kwargs["remove_swap"] = self.global_desc_remove_swapped_pairs
        if self.global_desc_num_refills_when_no_matches:
            kwargs["no_match_refills"] = self.global_desc_num_refills_when_no_matches
        if self.doppelgangers:
            kwargs["dg_th"] = self.doppelgangers.threshold
        param_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        return f"{self.type}({param_str})"


class GlobalDescriptorConfig(BaseModel):
    type: Literal[
        "apgem",
        "cvnet",
        "dinov2",
        "dinov2_salad",
        "patchnetvlad",
        "mast3r_spoc",
    ] = "dinov2"

    batch_size: int = 16
    num_workers: int = 2

    similar_distance_threshold: Optional[float] = None
    topk: Optional[int] = None

    apgem: Optional[APGeMConfig] = None
    cvnet: Optional[CVNetConfig] = None
    dinov2: Optional[HuggingFaceModelConfig] = None
    dinov2_salad: Optional[DINOv2SALADConfig] = None
    patchnetvlad: Optional[PatchNetVLADModelConfig] = None
    mast3r_spoc: Optional[MASt3RRetrievalModelConfig] = None

    def __str__(self) -> str:
        return self.type


class IMC2024ShortlistGeneratorConfig(BaseModel):
    global_descriptors: Optional[list[GlobalDescriptorConfig]] = None

    multiple_descs_fusion_type: Literal[
        "concat", "concat-and-normalize", "top-ranking"
    ] = "concat"

    similar_distance_threshold: Optional[float] = None
    topk: int = 20

    all_pairs_fallback_threshold: int = 20
    remove_swapped_pairs: bool = True  # Should be True
    num_refills_when_no_matches: int = 0
    compute_feats_if_fallback: bool = False

    compat_version: Literal["v2"]

    def __str__(self) -> str:
        descs = []
        if self.global_descriptors:
            descs = [str(d) for d in self.global_descriptors]
        return f"imc2024(descs={descs}, fusion={self.multiple_descs_fusion_type}, th={self.similar_distance_threshold}, topk={self.topk}, fallback={self.all_pairs_fallback_threshold}, remove_swap={self.remove_swapped_pairs}, no_match_refills={self.num_refills_when_no_matches})"


class LocalFeatureBasedANNShortlistUpdaterConfig(BaseModel):
    num_limit_index_feature: int = 8192
    num_limit_query_feature: int = 512
    k: int = 30
    dim: int = 256


class NoShortlistUpdaterConfig(BaseModel):
    global_descriptors: Optional[list[GlobalDescriptorConfig]] = None

    def __str__(self) -> str:
        gds = []
        if self.global_descriptors:
            for gd in self.global_descriptors:
                gds.append(gd.type)
        return f"noupdate(desc={gds})"


class PreMatchingShortlistUpdaterConfig(BaseModel):
    match_threshold: int
    ransac: Optional[RANSACConfig] = None
    global_descriptors: Optional[list[GlobalDescriptorConfig]] = None

    aggregation_type: Literal["union", "intersection"] = "union"

    def __str__(self) -> str:
        gds = []
        if self.global_descriptors:
            for gd in self.global_descriptors:
                gds.append(gd.type)
        return f"prematching(th={self.match_threshold}, ransac={self.ransac}, desc={gds}, agg={self.aggregation_type})"


class PreMatchingTopKShortlistUpdaterConfig(BaseModel):
    topk: int

    match_threshold: Optional[int] = None
    use_match_threshold: bool = False
    use_similarity_threshold: bool = False

    fallback_threshold: Optional[int] = None

    ransac: Optional[RANSACConfig] = None
    global_descriptors: Optional[list[GlobalDescriptorConfig]] = None

    path_length_for_additional_pairs: Optional[int] = None  # >1
    satisfied_edges_threshold: Optional[int] = None

    def __str__(self) -> str:
        gds = []
        if self.global_descriptors:
            for gd in self.global_descriptors:
                gds.append(gd.type)
        return f"prematching_topk(topk={self.topk}, ransac={self.ransac}, desc={gds}, length={self.path_length_for_additional_pairs}, satisfied_edges_th={self.satisfied_edges_threshold})"


class Line2DFeatureShortlistUpdaterConfig(BaseModel):
    theshold: float = 0.5


class ShortlistUpdaterConfig(BaseModel):
    type: str = "pre_matching"

    pre_matching: Optional[PreMatchingShortlistUpdaterConfig] = None
    pre_matching_topk: Optional[PreMatchingTopKShortlistUpdaterConfig] = None
    line2d_feature: Optional[Line2DFeatureShortlistUpdaterConfig] = None
    no_update: Optional[NoShortlistUpdaterConfig] = None

    def __str__(self) -> str:
        return str(getattr(self, self.type))
