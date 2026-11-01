"""Engine registry. Imports are lazy so a worker only loads the libraries it needs
(the fast engine never imports transformers or the PyTorch codec)."""
from __future__ import annotations

from .base import Engine, Request, Result, Voice


def create_engine(kind: str, settings: dict) -> Engine:
    if kind == "reference":
        from .reference import ReferenceEngine

        return ReferenceEngine(settings)
    if kind == "fast":
        from .fast import FastEngine

        return FastEngine(settings)
    if kind == "mock":
        from .mock import MockEngine

        return MockEngine(settings)
    raise ValueError(f"unknown engine {kind!r}; expected reference, fast or mock")


__all__ = ["Engine", "Request", "Result", "Voice", "create_engine"]
