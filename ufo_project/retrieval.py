from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalResult:
    question: str
    answer: str
    citations: pd.DataFrame
    elapsed: float
    used_chars: int


def answer_with_sources(df: pd.DataFrame, question: str, *, budget_chars: int = 1200, top_k: int = 5) -> RetrievalResult:
    started = time.perf_counter()
    work = df[df["comments"].str.strip() != ""].copy().reset_index().rename(columns={"index": "row_id"})
    vectorizer = TfidfVectorizer(max_features=12000, min_df=2, stop_words="english")
    matrix = vectorizer.fit_transform(work["comments"])
    query = vectorizer.transform([question])
    scores = cosine_similarity(query, matrix).ravel()
    order = np.argsort(scores)[::-1][:top_k]

    citations = work.iloc[order][["row_id", "datetime", "city", "state", "country", "shape", "comments"]].copy()
    citations["score_recherche"] = scores[order]
    packed_rows = []
    used_chars = 0
    for _, row in citations.iterrows():
        line = f"{row['row_id']} {row['datetime']} {row['city']} {row['shape']}: {row['comments']}"
        if used_chars + len(line) > budget_chars:
            break
        packed_rows.append(row)
        used_chars += len(line)

    if packed_rows:
        citations = pd.DataFrame(packed_rows)
    else:
        citations = citations.head(0)

    if citations.empty:
        answer = "Nous n'avons pas de relevé assez proche pour répondre sans inventer."
    else:
        shape_counts = citations["shape"].value_counts().head(3).to_dict()
        answer = (
            f"Les relevés retrouvés citent surtout ces formes : {shape_counts}. "
            f"Réponse fondée sur {len(citations)} relevés, pas sur une génération libre."
        )
    return RetrievalResult(question, answer, citations, time.perf_counter() - started, used_chars)


def naive_keyword_hits(df: pd.DataFrame, question: str, *, top_k: int = 5) -> int:
    words = {word.lower() for word in question.replace("?", " ").replace(",", " ").split() if len(word) > 3}
    if not words:
        return 0
    comments = df["comments"].fillna("").str.lower()
    mask = comments.map(lambda text: any(word in text for word in words))
    return int(mask.head(top_k).sum())
