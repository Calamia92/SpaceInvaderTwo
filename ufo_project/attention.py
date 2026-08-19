from __future__ import annotations

import hashlib

import numpy as np


def stable_token_vector(token: str, dim: int) -> np.ndarray:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, dim).astype(np.float32)


def sinusoidal_positions(length: int, dim: int) -> np.ndarray:
    positions = np.arange(length)[:, None]
    div = np.exp(np.arange(0, dim, 2) * (-np.log(10000.0) / dim))
    values = np.zeros((length, dim), dtype=np.float32)
    values[:, 0::2] = np.sin(positions * div)
    values[:, 1::2] = np.cos(positions * div)
    return values


def embed_tokens(tokens: list[str], dim: int = 24, *, positional: bool = False) -> np.ndarray:
    values = np.stack([stable_token_vector(token, dim) for token in tokens])
    if positional:
        values = values + sinusoidal_positions(len(tokens), dim)
    return values


def attention_forward(x: np.ndarray, seed: int = 10) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dim = x.shape[1]
    wq = rng.normal(0, 1 / np.sqrt(dim), (dim, dim))
    wk = rng.normal(0, 1 / np.sqrt(dim), (dim, dim))
    wv = rng.normal(0, 1 / np.sqrt(dim), (dim, dim))

    q = x @ wq
    k = x @ wk
    v = x @ wv
    scores = q @ k.T / np.sqrt(dim)
    scores = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights @ v, weights
