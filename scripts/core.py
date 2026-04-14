from __future__ import annotations

import contextlib
import io
import time
from pathlib import Path
from typing import Literal, Self

import pydantic
import torch
import yaml


class ConfigBlock(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(protected_namespaces=())

    @classmethod
    def from_yaml_string(cls: type[Self], text: str) -> Self:
        return cls.model_validate(yaml.safe_load(io.BytesIO(text.encode())))

    @classmethod
    def from_file(
        cls: type[Self], path: str | Path, file_format: Literal["yaml"] = "yaml"
    ) -> Self:
        if file_format == "yaml":
            return cls.model_validate(yaml.safe_load(Path(path).read_text()))
        raise ValueError(file_format)


@contextlib.contextmanager
def perf_time(title: str, cuda: bool = False):
    start_at = time.perf_counter()
    try:
        yield
    finally:
        if cuda:
            torch.cuda.synchronize()
        end_at = time.perf_counter()
    elapsed = end_at - start_at
    print(f"({title}) Elapsed time: {elapsed}")
