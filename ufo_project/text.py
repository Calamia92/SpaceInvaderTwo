from __future__ import annotations

import re

import pandas as pd

TOKEN_RE = re.compile(r"[a-z0-9']+")
SHAPE_MERGES = {
    "round": "circle",
    "changed": "changing",
}
FOUR_KEYWORDS = {"unknown", "other"}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def normalize_shape(shape: str) -> str:
    value = str(shape).strip().lower()
    return SHAPE_MERGES.get(value, value)


def clean_shape_rows(df: pd.DataFrame, *, keep_fourre_tout: bool = False) -> pd.DataFrame:
    work = df[["comments", "shape", "observed_at"]].copy()
    work["comments"] = work["comments"].fillna("").astype(str)
    work["shape"] = work["shape"].map(normalize_shape)
    work = work[(work["shape"] != "") & (work["comments"].str.strip() != "")]
    if not keep_fourre_tout:
        work = work[~work["shape"].isin(FOUR_KEYWORDS)]
    return work.reset_index(drop=True)
