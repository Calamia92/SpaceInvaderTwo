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


@dataclass
class RetrievalSystemMeasurement:
    name: str
    max_features: int
    index_bytes: int
    build_seconds: float
    latency_seconds: float
    throughput_qps: float
    sourced_answers: int
    mean_overlap_with_reference: float


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


def _matrix_bytes(matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def measure_retrieval_systems(
    df: pd.DataFrame,
    questions: list[str],
    *,
    budget_chars: int = 1200,
    top_k: int = 6,
) -> list[RetrievalSystemMeasurement]:
    work = df[df["comments"].str.strip() != ""].copy().reset_index().rename(columns={"index": "row_id"})
    systems = [("avant", 12000), ("après", 4000)]
    reference_ids: list[set[int]] | None = None
    measurements: list[RetrievalSystemMeasurement] = []

    for name, max_features in systems:
        started_build = time.perf_counter()
        vectorizer = TfidfVectorizer(max_features=max_features, min_df=2, stop_words="english")
        matrix = vectorizer.fit_transform(work["comments"])
        build_seconds = time.perf_counter() - started_build
        index_bytes = _matrix_bytes(matrix) + sum(len(token) for token in vectorizer.vocabulary_)

        started_queries = time.perf_counter()
        sourced = 0
        retrieved_sets: list[set[int]] = []
        for question in questions:
            query = vectorizer.transform([question])
            scores = cosine_similarity(query, matrix).ravel()
            order = np.argsort(scores)[::-1][:top_k]
            citations = work.iloc[order][["row_id", "datetime", "city", "shape", "comments"]].copy()
            packed: list[int] = []
            used_chars = 0
            for _, row in citations.iterrows():
                line = f"{row['row_id']} {row['datetime']} {row['city']} {row['shape']}: {row['comments']}"
                if used_chars + len(line) > budget_chars:
                    break
                packed.append(int(row["row_id"]))
                used_chars += len(line)
            sourced += int(bool(packed))
            retrieved_sets.append(set(packed))
        latency_seconds = (time.perf_counter() - started_queries) / max(len(questions), 1)
        throughput_qps = 1 / max(latency_seconds, 1e-9)

        if reference_ids is None:
            overlap = 1.0
            reference_ids = retrieved_sets
        else:
            scores = []
            for current, reference in zip(retrieved_sets, reference_ids):
                if not current and not reference:
                    scores.append(1.0)
                elif not current or not reference:
                    scores.append(0.0)
                else:
                    scores.append(len(current & reference) / len(current | reference))
            overlap = float(np.mean(scores))

        measurements.append(
            RetrievalSystemMeasurement(
                name=name,
                max_features=max_features,
                index_bytes=index_bytes,
                build_seconds=build_seconds,
                latency_seconds=latency_seconds,
                throughput_qps=throughput_qps,
                sourced_answers=sourced,
                mean_overlap_with_reference=overlap,
            )
        )
    return measurements
