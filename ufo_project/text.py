from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

TOKEN_RE = re.compile(r"[a-z0-9']+")
SHAPE_MERGES = {
    "round": "circle",
    "changed": "changing",
}
FOUR_KEYWORDS = {"unknown", "other"}


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    classes: list[str]
    kept_rows: int
    decisions: str


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


def prepare_shape_dataset(
    df: pd.DataFrame,
    *,
    min_class_count: int = 300,
    keep_fourre_tout: bool = False,
    random_state: int = 7,
) -> DatasetSplit:
    work = clean_shape_rows(df, keep_fourre_tout=keep_fourre_tout)
    counts = work["shape"].value_counts()
    allowed = counts[counts >= min_class_count].index
    work = work[work["shape"].isin(allowed)].reset_index(drop=True)

    classes = sorted(work["shape"].unique())
    train_valid, test = train_test_split(
        work,
        test_size=0.15,
        stratify=work["shape"],
        random_state=random_state,
    )
    train, valid = train_test_split(
        train_valid,
        test_size=0.1765,
        stratify=train_valid["shape"],
        random_state=random_state,
    )
    decisions = (
        "Les relevés sans forme sont supprimés car ils ne fournissent aucune cible vérifiable. "
        "`unknown` et `other` sont supprimés car ce sont des fourre-tout. "
        "`round` est fusionné avec `circle` et `changed` avec `changing`. "
        f"Une classe est gardée seulement à partir de {min_class_count} relevés."
    )
    return DatasetSplit(
        train=train.reset_index(drop=True),
        valid=valid.reset_index(drop=True),
        test=test.reset_index(drop=True),
        classes=classes,
        kept_rows=len(work),
        decisions=decisions,
    )


def banned_shape_words(classes: list[str]) -> set[str]:
    words: set[str] = set()
    for label in classes:
        for token in tokenize(label):
            words.add(token)
            words.add(f"{token}s")
            if token.endswith("y"):
                words.add(f"{token[:-1]}ies")
    for source, target in SHAPE_MERGES.items():
        for token in [source, target]:
            words.add(token)
            words.add(f"{token}s")
    words.update({"disc", "discs", "disk", "disks", "circular", "triangular", "sphere", "spheres"})
    return words


def remove_banned_words(text: str, banned: set[str]) -> str:
    return " ".join("[FORME]" if token in banned else token for token in tokenize(text))


def count_rows_with_banned_words(texts, banned: set[str]) -> int:
    total = 0
    for text in texts:
        if any(token in banned for token in tokenize(text)):
            total += 1
    return total
