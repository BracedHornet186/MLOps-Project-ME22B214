from __future__ import annotations

import torch

from retrievers.base import Retriever
from retrievers.config import RetrieverConfig
from retrievers.mast3r_retrieval_asmk import MASt3RRetrievalASMKRetriever


def create_retriever(
    conf: RetrieverConfig, device: torch.device | None = None
) -> Retriever:
    if conf.type == "mast3r_retrieval_asmk":
        assert conf.mast3r_retrieval_asmk
        retriever = MASt3RRetrievalASMKRetriever(
            conf.mast3r_retrieval_asmk,
            device=device,
        )
    else:
        raise ValueError(conf.type)
    return retriever
