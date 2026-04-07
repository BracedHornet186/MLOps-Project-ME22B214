from __future__ import annotations

from typing import Optional, Protocol

from core import ConfigBlock
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


class GlobalDescriptorExtractorConfigProtocol(Protocol):
    global_desc_model: str | None

    apgem: Optional[APGeMConfig] = None
    cvnet: Optional[CVNetConfig] = None
    dinov2: Optional[HuggingFaceModelConfig] = None
    dinov2_salad: Optional[DINOv2SALADConfig] = None
    patchnetvlad: Optional[PatchNetVLADModelConfig] = None
    mast3r_spoc: Optional[MASt3RRetrievalModelConfig] = None
    mast3r_retrieval_spoc: Optional[MASt3RRetrievalModelConfig] = None
    moge: Optional[MoGeModelConfig] = None
    moge_depth_hog: Optional[MoGeModelConfig] = None
    moge_dinov2: Optional[MoGeModelConfig] = None
    siglip2: Optional[HuggingFaceModelConfig] = None
    isc: Optional[ISCModelConfig] = None


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
    moge_depth_hog: Optional[MoGeModelConfig] = None
    moge_dinov2: Optional[MoGeModelConfig] = None
    siglip2: Optional[HuggingFaceModelConfig] = None
    isc: Optional[ISCModelConfig] = None
