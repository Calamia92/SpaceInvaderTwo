from __future__ import annotations

import hashlib

import numpy as np


def stable_token_vector(token: str, dim: int) -> np.ndarray:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, dim).astype(np.float32)


def embed_tokens(tokens: list[str], dim: int = 24) -> np.ndarray:
    return np.stack([stable_token_vector(token, dim) for token in tokens])


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
