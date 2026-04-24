from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import cv2
import mast3r.retrieval.model
import mast3r.retrieval.processor
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from dust3r.utils.image import load_images
from mast3r.model import AsymmetricMASt3R
from torch.utils.data.dataset import Dataset

from scripts.data import FilePath, resolve_model_path
from models.mast3r.model import get_mast3r_model
from pipelines.scene import Scene
from retrievers.base import Retriever
from retrievers.config import MAST3RRetrievalASMKRetrieverConfig


class MASt3RRetrievalASMKRetriever(Retriever):
    def __init__(
        self,
        conf: MAST3RRetrievalASMKRetrieverConfig,
        device: torch.device | None = None,
    ):
        self.conf = conf
        self.device = device or torch.device("cuda")
        backbone = get_mast3r_model(
            resolve_model_path(conf.mast3r.weight_path), self.device
        )
        assert conf.mast3r.retrieval_weight_path and conf.mast3r.retrieval_codebook_path
        self.retriever = mast3r.retrieval.processor.Retriever(
            str(resolve_model_path(conf.mast3r.retrieval_weight_path)),
            backbone=backbone,
            device=self.device,  # type: ignore
        )
        self.imsize = 512
        self.db = None

    def build(self, image_paths: list[str | Path]) -> MASt3RRetrievalASMKRetriever:
        feat, ids = mast3r.retrieval.model.extract_local_features(
            self.retriever.model,
            image_paths,
            self.imsize,
            tocpu=True,
            device=self.device,
        )
        feat = feat.cpu().numpy()
        ids = ids.cpu().numpy()

        # NOTE
        # When len(image_paths) == 2
        #  feat.shape: (600, 1024)
        #  ids.shape: (600,)
        #  print(ids): [0, 0, 0, ..., 0, 1, 1, 1, ..., 1]

        # Init
        print(f"Building IVF based on feat({len(feat)})")
        self.db = self.retriever.asmk.build_ivf(feat, ids)
        self.feat = feat
        self.ids = ids
        print("Building IVF: Done")

        return self

    def search_by_id(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.db is not None
        feats = self.feat[self.ids == i]
        ids = self.ids[self.ids == i]
        metadata, _, ranks, sims = self.db.query_ivf(feats, ids)
        assert len(ranks) == 1
        assert len(sims) == 1
        return ranks[0], sims[0]

    def search_nn(
        self,
        image_paths: list[str | Path],
        k: int | None = None,
        return_metric: Literal["dist", "sim"] = "sim",
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.db is not None
        ranks_list = []
        sims_list = []
        for i in range(len(image_paths)):
            _ranks, _sims = self.search_by_id(i)
            # print(_ranks, _sims)
            ranks_list.append(_ranks)
            sims_list.append(_sims.astype(np.float32))

        if len(ranks_list) == 0:
            return np.empty((0, 0), dtype=np.int64), np.empty((0, 0), dtype=np.float32)

        ranks = np.vstack(ranks_list)
        sims = np.vstack(sims_list)

        if k is not None:
            ranks = ranks[:, :k]
            sims = sims[:, :k]

        if return_metric == "dist":
            return ranks, 1 - sims

        return ranks, sims

    def get_sim_matrix(self, image_paths: list[str | Path]) -> np.ndarray:
        ranks, _sims = self.search_nn(image_paths, return_metric="sim")
        sims = np.empty_like(_sims)
        sims[np.arange(_sims.shape[0])[:, None], ranks] = _sims
        return sims
