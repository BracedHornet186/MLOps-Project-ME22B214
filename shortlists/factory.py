from __future__ import annotations

from typing import Optional

import torch

from shortlists.base import (
    DebugShortlistGenerator,
    ShortlistGenerator,
    ShortlistUpdater,
)
from shortlists.config import (
    IMC2024ShortlistGeneratorConfig,
    ShortlistGeneratorConfig,
    ShortlistUpdaterConfig,
)
from shortlists.connected_component import (
    GlobalDescriptorConnectedComponentShortlistGenerator,
)
from shortlists.cvnet_rerank import CVNetRerankShortlistGenerator
from shortlists.doppelgangers import DoppelGangersShortlistGenerator
from shortlists.ensemble import EnsembleShortlistGenerator
from shortlists.exhaustive import AllPairsShortlistGenerator
from shortlists.global_descriptor import GlobalDescriptorShortlistGenerator
from shortlists.global_descriptor_fps import GlobalDescriptorFPSShortlistGenerator
from shortlists.imc2024 import IMC2024ShortlistGenerator
from shortlists.line2d_feature import Line2DFeatureShortlistUpdater
from shortlists.mast3r_retrieval_asmk import MASt3RRetrievalASMKShortlistGenerator
from shortlists.no_update import NoShortlistUpdater
from shortlists.prematching import PreMatchingShortlistUpdater
from shortlists.prematching_topk import PreMatchingTopKShortlistUpdater
from shortlists.random_walk import PreMatchingRandomWalkShortlistGenerator


def create_shortlist_generator(
    conf: ShortlistGeneratorConfig | IMC2024ShortlistGeneratorConfig,
    device: Optional[torch.device] = None,
) -> ShortlistGenerator:
    if isinstance(conf, IMC2024ShortlistGeneratorConfig):
        return IMC2024ShortlistGenerator(conf, device=device)

    if conf.type == "ensemble":
        assert conf.ensemble
        generator = EnsembleShortlistGenerator(
            conf.ensemble,
            [
                create_shortlist_generator(c, device=device)
                for c in conf.ensemble.shortlist_generators
            ],
        )
    elif conf.type == "all_pairs":
        generator = AllPairsShortlistGenerator()
    elif conf.type == "debug":
        print("[create_shortlist_generator] DEBUG mode!!!!!")
        generator = DebugShortlistGenerator()
    elif conf.type == "gdcc":
        generator = GlobalDescriptorConnectedComponentShortlistGenerator(
            conf, device=device
        )
    elif conf.type == "global_desc":
        generator = GlobalDescriptorShortlistGenerator(conf, device=device)
    elif conf.type == "global_desc_fps":
        generator = GlobalDescriptorFPSShortlistGenerator(conf, device=device)
    elif conf.type == "cvnet_rerank":
        generator = CVNetRerankShortlistGenerator(conf, device=device)
    elif conf.type == "random_walk":
        generator = PreMatchingRandomWalkShortlistGenerator(conf)
    elif conf.type == "doppelgangers":
        generator = DoppelGangersShortlistGenerator(conf, device=device)
    elif conf.type == "mast3r_retrieval_asmk":
        generator = MASt3RRetrievalASMKShortlistGenerator(conf, device=device)
    else:
        raise ValueError(conf.type)
    return generator


def create_shortlist_updater(
    conf: ShortlistUpdaterConfig, device: Optional[torch.device] = None
) -> ShortlistUpdater:
    if conf.type == "pre_matching":
        assert conf.pre_matching
        updater = PreMatchingShortlistUpdater(conf.pre_matching, device=device)
    elif conf.type == "pre_matching_topk":
        assert conf.pre_matching_topk
        updater = PreMatchingTopKShortlistUpdater(conf.pre_matching_topk, device=device)
    elif conf.type == "line2d_feature":
        assert conf.line2d_feature
        updater = Line2DFeatureShortlistUpdater(conf.line2d_feature, device=device)
    elif conf.type == "no_update":
        assert conf.no_update
        updater = NoShortlistUpdater(conf.no_update, device=device)
    else:
        raise ValueError(conf.type)
    return updater
