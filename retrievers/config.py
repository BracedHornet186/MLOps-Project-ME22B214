from __future__ import annotations

from typing import Literal, Optional, TypeAlias

from core import ConfigBlock
from models.config import MASt3RModelConfig

RetrieverType: TypeAlias = Literal["mast3r_retrieval_asmk"]


class MAST3RRetrievalASMKRetrieverConfig(ConfigBlock):
    mast3r: MASt3RModelConfig


class RetrieverConfig(ConfigBlock):
    type: RetrieverType

    mast3r_retrieval_asmk: Optional[MAST3RRetrievalASMKRetrieverConfig] = None
