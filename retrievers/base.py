from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import numpy as np


class Retriever:
    def build(self: Self, image_paths: list[str | Path]) -> Self: ...

    def search_by_id(self, i: int) -> tuple[np.ndarray, np.ndarray]: ...

    def search_nn(
        self,
        image_paths: list[str | Path],
        k: int | None = None,
        return_metric: Literal["dist", "sim"] = "sim",
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def get_sim_matrix(self, image_paths: list[str | Path]) -> np.ndarray: ...
