from pydantic import BaseModel
from typing import Literal, Optional

from matchers.config import DetectorFreeMatcherConfig
from postprocesses.config import RANSACConfig
from models.config import MicKeyModelConfig


class MapFreeConfig(BaseModel):
    matcher: DetectorFreeMatcherConfig
    pose_solver: str = "pnp"


class MicKeyConfig(BaseModel):
    model: MicKeyModelConfig


class TwoViewLocalizerConfig(BaseModel):
    matcher: DetectorFreeMatcherConfig
    ransac: RANSACConfig


class LocalizerConfig(BaseModel):
    type: Literal["mapfree", "mickey"]

    mapfree: Optional[MapFreeConfig] = None
    mickey: Optional[MicKeyConfig] = None


class PostLocalizerConfig(BaseModel):
    type: Literal["mapfree", "two_view"]

    mapfree: Optional[MapFreeConfig] = None
    two_view: Optional[TwoViewLocalizerConfig] = None