from __future__ import annotations

from typing import Optional

import torch

from scripts.extractor import LocalFeatureExtractor
from features.factory import create_local_feature_handler
from matchers._research_only.aspanformer import ASpanFormerMatcher
from matchers._research_only.magicleap_superglue import MagicLeapSuperGlueMatcher
from matchers._research_only.magicleap_superglue_rot import (
    MagicLeapSuperGlueRotationMatcher,
)
from matchers.adamatcher import AdaMatcherMatcher
from matchers.base import (
    DetectorFreeMatcher,
    Line2DFeatureMatcher,
    LocalFeatureMatcher,
    PointTrackingMatcher,
)
from matchers.config import (
    DetectorFreeMatcherConfig,
    Line2DFeatureMatcherConfig,
    LocalFeatureMatcherConfig,
    PointTrackingMatcherConfig,
)
from matchers.dkm import DKMMatcher
from matchers.dkm_rot import DKMRotationMatcher
from matchers.dual_softmax import DualSoftmaxMatcher
from matchers.efficientloftr import EfficientLoFTRMatcher
from matchers.gim_dkm import GIMDKMMatcher
from matchers.gim_lightglue import GIMLightGlueMatcher
from matchers.gim_loftr import GIMLoFTRMatcher
from matchers.lightglue import LightGlueMatcher
from matchers.loftr import LoFTRMatcher
from matchers.mast3r import MASt3RMatcher
from matchers.mast3r_c2f import MASt3RC2FMatcher
from matchers.mast3r_hybrid import MASt3RHybridMatcher
from matchers.mast3r_mpsfm_sparse import MASt3RMPSFMSparseMatcher
from matchers.mast3r_sparse import MASt3RSparseMatcher
from matchers.matchformer import MatchformerMatcher
from matchers.nn_double_softmax import NNDoubleSoftmaxMatcher
from matchers.nn_matcher import NNMatcher
from matchers.omniglue_onnx import OmniGlueONNXMatcher
from matchers.openglue import OpenGlueMatcher
from matchers.roma import RoMaMatcher
from matchers.se2loftr import SE2LoFTRMatcher
from matchers.steerer import SteererMatcher
from matchers.vggt import CustomVGGTMatcher, VGGTMatcher
from matchers.xfeat import XFeatStarMatcher
from postprocesses.panet import PANetRefiner

# from matchers.quadtree import QuadTreeMatcher


def create_local_feature_matcher(
    conf: LocalFeatureMatcherConfig, device: Optional[torch.device] = None
) -> LocalFeatureMatcher:
    refiner = None
    if conf.panet:
        refiner = PANetRefiner(conf.panet, device=device)
        print(f"Use refiner in matcher({conf.type})")

    if conf.type == "magicleap_superglue":
        assert conf.magicleap_superglue
        matcher = MagicLeapSuperGlueMatcher(
            conf.magicleap_superglue, refiner=refiner, device=device
        )
    elif conf.type == "nn":
        assert conf.nn
        matcher = NNMatcher(conf.nn, refiner=refiner, device=device)
    elif conf.type == "nn_double_softmax":
        assert conf.nn_double_softmax
        matcher = NNDoubleSoftmaxMatcher(
            conf.nn_double_softmax, refiner=refiner, device=device
        )
    elif conf.type == "openglue":
        assert conf.openglue
        matcher = OpenGlueMatcher(conf.openglue, refiner=refiner, device=device)
    elif conf.type == "gim_lightglue":
        assert conf.gim_lightglue
        matcher = GIMLightGlueMatcher(
            conf.gim_lightglue, refiner=refiner, device=device
        )
    elif conf.type == "lightglue":
        assert conf.lightglue
        matcher = LightGlueMatcher(conf.lightglue, refiner=refiner, device=device)
    elif conf.type == "dual_softmax":
        assert conf.dual_softmax
        matcher = DualSoftmaxMatcher(conf.dual_softmax, refiner=refiner, device=device)
    elif conf.type == "steerer":
        assert conf.steerer
        matcher = SteererMatcher(conf.steerer, refiner=refiner, device=device)
    else:
        raise ValueError(conf.type)
    return matcher


def create_detector_free_matcher(
    conf: DetectorFreeMatcherConfig, device: Optional[torch.device] = None
) -> DetectorFreeMatcher:
    if conf.type == "adamatcher":
        assert conf.adamatcher
        matcher = AdaMatcherMatcher(conf.adamatcher, device=device)
    elif conf.type == "aspanformer":
        assert conf.aspanformer
        matcher = ASpanFormerMatcher(conf.aspanformer, device=device)
    elif conf.type == "dkm":
        assert conf.dkm
        matcher = DKMMatcher(conf.dkm, device=device)
    elif conf.type == "dkm_rotation":
        assert conf.dkm_rotation
        matcher = DKMRotationMatcher(conf.dkm_rotation, device=device)
    elif conf.type == "ecotr":
        assert conf.ecotr
        raise NotImplementedError("Not available: type=ecotr (Required: pykeops)")
    elif conf.type == "efficientloftr":
        assert conf.efficientloftr
        matcher = EfficientLoFTRMatcher(conf.efficientloftr, device=device)
    elif conf.type == "gim_dkm":
        assert conf.gim_dkm
        matcher = GIMDKMMatcher(conf.gim_dkm, device=device)
    elif conf.type == "gim_loftr":
        assert conf.gim_loftr
        matcher = GIMLoFTRMatcher(conf.gim_loftr, device=device)
    elif conf.type == "loftr":
        assert conf.loftr
        matcher = LoFTRMatcher(conf.loftr, device=device)
    elif conf.type == "magicleap_superglue_rotation":
        assert conf.magicleap_superglue_rotation
        matcher = MagicLeapSuperGlueRotationMatcher(
            conf.magicleap_superglue_rotation, device=device
        )
    elif conf.type == "mast3r":
        assert conf.mast3r
        matcher = MASt3RMatcher(conf.mast3r, device=device)
    elif conf.type == "mast3r_c2f":
        assert conf.mast3r_c2f
        matcher = MASt3RC2FMatcher(conf.mast3r_c2f, device=device)
    elif conf.type == "matchformer":
        assert conf.matchformer
        matcher = MatchformerMatcher(conf.matchformer, device=device)
    elif conf.type == "omniglue_onnx":
        assert conf.omniglue_onnx
        matcher = OmniGlueONNXMatcher(conf.omniglue_onnx, device=device)
    elif conf.type == "quadtree":
        assert conf.quadtree
        raise NotImplementedError
        # matcher = QuadTreeMatcher(conf.quadtree, device=device)
    elif conf.type == "roma":
        assert conf.roma
        matcher = RoMaMatcher(conf.roma, device=device)
    elif conf.type == "se2loftr":
        assert conf.se2loftr
        matcher = SE2LoFTRMatcher(conf.se2loftr, device=device)
    elif conf.type == "xfeat_star":
        assert conf.xfeat_star
        matcher = XFeatStarMatcher(conf.xfeat_star, device=device)
    else:
        raise ValueError(conf.type)
    return matcher


def create_line2d_feature_matcher(
    conf: Line2DFeatureMatcherConfig, device: Optional[torch.device] = None
) -> Line2DFeatureMatcher:
    if conf.type == "limap":
        from matchers.line2d import LIMAPMatcher

        assert conf.limap
        matcher = LIMAPMatcher(conf.limap)
    else:
        raise ValueError(conf.type)
    return matcher


def create_point_tracking_matcher(
    conf: PointTrackingMatcherConfig,
    device: torch.device | None = None,
) -> PointTrackingMatcher:
    if conf.type == "vggt":
        assert conf.vggt
        extractors = []
        for c in conf.local_features:
            handler = create_local_feature_handler(c, device=device)
            extractor = LocalFeatureExtractor(c, handler)
            extractors.append(extractor)
        device = device or torch.device("cuda")
        if conf.vggt.use_custom_vggt:
            matcher = CustomVGGTMatcher(conf.vggt, extractors, device)
        else:
            matcher = VGGTMatcher(conf.vggt, extractors, device)
    elif conf.type == "mast3r_mpsfm_sparse":
        assert conf.mast3r_mpsfm_sparse
        extractors = []
        for c in conf.local_features:
            handler = create_local_feature_handler(c, device=device)
            extractor = LocalFeatureExtractor(c, handler)
            extractors.append(extractor)
        device = device or torch.device("cuda")
        matcher = MASt3RMPSFMSparseMatcher(conf.mast3r_mpsfm_sparse, extractors, device)
    elif conf.type == "mast3r_sparse":
        assert conf.mast3r_sparse
        extractors = []
        for c in conf.local_features:
            handler = create_local_feature_handler(c, device=device)
            extractor = LocalFeatureExtractor(c, handler)
            extractors.append(extractor)
        device = device or torch.device("cuda")
        matcher = MASt3RSparseMatcher(conf.mast3r_sparse, extractors, device)
    elif conf.type == "mast3r_hybrid":
        assert conf.mast3r_hybrid
        extractors = []
        for c in conf.local_features:
            handler = create_local_feature_handler(c, device=device)
            print(f"MASt3RHybridMatcher | local_features={handler.__class__.__name__}")
            extractor = LocalFeatureExtractor(c, handler)
            extractors.append(extractor)
        device = device or torch.device("cuda")
        matcher = MASt3RHybridMatcher(conf.mast3r_hybrid, extractors, device)
    else:
        raise ValueError(conf.type)
    return matcher
