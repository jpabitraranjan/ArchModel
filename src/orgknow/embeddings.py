"""Embedding interface. Swap in a real model (OpenAI, Voyage, sentence-transformers, ...) by
implementing ``embed``. The default is a dependency-free hashing embedder: good enough for
tests and lexical-ish similarity, NOT a semantic model."""
from __future__ import annotations

import math
import re
import zlib
from typing import Protocol


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    name = "hashing-256"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        toks = re.findall(r"[a-z0-9]+", text.lower())
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = [0.0] * self.dim
        for g in grams:
            h = zlib.crc32(g.encode())
            v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
