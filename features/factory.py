from __future__ import annotations

from typing import Optional

import torch

from features._research_only.magicleap_superpoint import MagicLeapSuperPointHandler
from features._research_only.silk import SiLKHandler
from features.alike import ALIKEHandler
from features.base import Line2DFeatureHandler, LocalFeatureHandler
from features.config import Line2DFeatureConfig, LocalFeatureConfig
from features.dad import DaDHandler
from features.darkfeat import DarkFeatHandler
from features.dedode import DeDoDeHandler
from features.delf_pytorch import DELFPytorchFeatHandler
from features.disk import DISKHandler
from features.edgepoint2 import EdgePoint2Handler
from features.featurebooster import FeatureBoosterHandler
from features.gim.superpoint import GIMSuperPointHandler
from features.hardnet import DoGAffNetHardNetHandler, KeyNetAffNetHardNetHandler
from features.hardnet8 import KeyNetAffNetHardNet8Handler
from features.lanet import LANetHandler
from features.lightglue.aliked import LightGlueALIKEDHandler
from features.lightglue.aliked_dedode_v2 import LightGlueALIKEDDeDoDeV2Handler
from features.lightglue.aliked_tile import LightGlueALIKEDTileHandler
from features.lightglue.disk import LightGlueDISKHandler
from features.lightglue.doghardnet import LightGlueDoGHardNetHandler
from features.lightglue.sift import LightGlueSIFTHandler
from features.lightglue.superpoint import LightGlueSuperPointHandler
from features.mtldesc import MTLDescHandler
from features.multiscale import FeatureSetHandler
from features.posfeat import PosFeatHandler
from features.relf import RELFHandler
from features.sfd2 import SFD2Handler
from features.superpoint import SuperPointHandler
from features.xfeat import XFeatHandler


def create_local_feature_handler(
    conf: LocalFeatureConfig, device: Optional[torch.device] = None
) -> LocalFeatureHandler:
    """Factory method of local feature handler"""
    if conf.type == "alike":
        assert conf.alike
        handler = ALIKEHandler(conf.alike, device=device)
    elif conf.type == "aliked":
        # from features.aliked import ALIKEDHandler

        assert conf.aliked
        raise NotImplementedError(
            "Not available: type=aliked (Use type=lightglue_aliked instead)"
        )
    elif conf.type == "lightglue_aliked":
        assert conf.lightglue_aliked
        handler = LightGlueALIKEDHandler(conf.lightglue_aliked, device=device)
    elif conf.type == "lightglue_aliked_dedode_v2":
        assert conf.lightglue_aliked_dedode_v2
        handler = LightGlueALIKEDDeDoDeV2Handler(
            conf.lightglue_aliked_dedode_v2, device=device
        )
    elif conf.type == "lightglue_aliked_tile":
        assert conf.lightglue_aliked_tile
        handler = LightGlueALIKEDTileHandler(conf.lightglue_aliked_tile, device=device)
    elif conf.type == "dad":
        assert conf.dad
        handler = DaDHandler(conf.dad, device=device)
    elif conf.type == "darkfeat":
        assert conf.darkfeat
        handler = DarkFeatHandler(conf.darkfeat, device=device)
    elif conf.type == "dedode":
        assert conf.dedode
        handler = DeDoDeHandler(conf.dedode, device=device)
    elif conf.type == "delf_pytorch":
        assert conf.delf_pytorch
        handler = DELFPytorchFeatHandler(conf.delf_pytorch, device=device)
    elif conf.type == "disk":
        assert conf.disk
        handler = DISKHandler(conf.disk, device=device)
    elif conf.type == "edgepoint2":
        assert conf.edgepoint2
        handler = EdgePoint2Handler(conf.edgepoint2, device=device)
    elif conf.type == "featurebooster":
        assert conf.featurebooster
        handler = FeatureBoosterHandler(conf.featurebooster, device=device)
    elif conf.type == "featureset":
        assert conf.featureset
        handler = FeatureSetHandler(
            [
                create_local_feature_handler(c, device=device)
                for c in conf.featureset.local_features
            ],
            conf.featureset,
            device=device,
        )
    elif conf.type == "gim_superpoint":
        assert conf.gim_superpoint
        handler = GIMSuperPointHandler(conf.gim_superpoint, device=device)
    elif conf.type == "hardnet":
        assert conf.hardnet
        if conf.hardnet.detector == "keynet":
            handler = KeyNetAffNetHardNetHandler(conf.hardnet, device=device)
        elif conf.hardnet.detector == "sift":
            handler = DoGAffNetHardNetHandler(conf.hardnet, device=device)
        else:
            raise ValueError
    elif conf.type == "hardnet8":
        assert conf.hardnet8
        if conf.hardnet8.detector == "keynet":
            handler = KeyNetAffNetHardNet8Handler(conf.hardnet8, device=device)
        elif conf.hardnet8.detector == "sift":
            raise NotImplementedError
        else:
            raise ValueError
    elif conf.type == "lanet":
        assert conf.lanet
        handler = LANetHandler(conf.lanet, device=device)
    elif conf.type == "lightglue_disk":
        assert conf.lightglue_disk
        handler = LightGlueDISKHandler(conf.lightglue_disk, device=device)
    elif conf.type == "lightglue_doghardnet":
        assert conf.lightglue_doghardnet
        handler = LightGlueDoGHardNetHandler(conf.lightglue_doghardnet, device=device)
    elif conf.type == "lightglue_sift":
        assert conf.lightglue_sift
        handler = LightGlueSIFTHandler(conf.lightglue_sift, device=device)
    elif conf.type == "lightglue_superpoint":
        assert conf.lightglue_superpoint
        handler = LightGlueSuperPointHandler(conf.lightglue_superpoint, device=device)
    elif conf.type == "magicleap_superpoint":
        assert conf.magicleap_superpoint
        handler = MagicLeapSuperPointHandler(conf.magicleap_superpoint, device=device)
    elif conf.type == "mtldesc":
        assert conf.mtldesc
        handler = MTLDescHandler(conf.mtldesc, device=device)
    elif conf.type == "multiscale":
        # Compatible
        assert conf.multiscale
        handler = FeatureSetHandler(
            [
                create_local_feature_handler(c, device=device)
                for c in conf.multiscale.local_features
            ],
            conf.multiscale,
            device=device,
        )
    elif conf.type == "posfeat":
        assert conf.posfeat
        handler = PosFeatHandler(conf.posfeat, device=device)
    elif conf.type == "relf":
        from scripts.extractor import LocalFeatureExtractor

        assert conf.relf
        detector = LocalFeatureExtractor(
            conf.relf.detector,
            create_local_feature_handler(conf.relf.detector, device=device),
        )
        handler = RELFHandler(conf.relf, detector, device=device)
    elif conf.type == "sfd2":
        assert conf.sfd2
        handler = SFD2Handler(conf.sfd2, device=device)
    elif conf.type == "silk":
        assert conf.silk
        handler = SiLKHandler(conf.silk, device=device)
    elif conf.type == "superpoint":
        assert conf.superpoint
        handler = SuperPointHandler(conf.superpoint, device=device)
    elif conf.type == "xfeat":
        assert conf.xfeat
        handler = XFeatHandler(conf.xfeat, device=device)
    else:
        raise ValueError(conf.type)
    return handler


def create_line2d_feature_handler(
    conf: Line2DFeatureConfig, device: Optional[torch.device] = None
) -> Line2DFeatureHandler:
    if conf.type == "limap":
        from features.line2d import LIMAPDetectionHandler

        assert conf.limap
        handler = LIMAPDetectionHandler(conf.limap)
    else:
        raise ValueError(conf.type)
    return handler
