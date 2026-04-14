"""
core/__init__.py — Base configuration block used by all pipeline config classes.
"""
from __future__ import annotations
from pydantic import BaseModel


class ConfigBlock(BaseModel):
    """Base configuration class. All pipeline sub-configs inherit from this."""
    model_config = {"extra": "allow", "populate_by_name": True}

    def __str__(self) -> str:
        fields = ", ".join(f"{k}={v}" for k, v in self.__dict__.items() if v is not None)
        return f"{self.__class__.__name__}({fields})"
