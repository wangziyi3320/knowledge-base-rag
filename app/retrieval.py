from __future__ import annotations

import math
import re
from collections import Counter

from app.embeddings import tokenize


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    tokenized = [tokenize(document) for document in documents]
    query_terms = tokenize(query)
    if not tokenized or not query_terms:
        return [0.0] * len(documents)
    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized) or 1.0
    document_frequency = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    total = len(tokenized)
    scores = []
    for tokens in tokenized:
        frequencies = Counter(tokens)
        length = len(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            frequency_in_docs = document_frequency[term]
            inverse = math.log(1 + (total - frequency_in_docs + 0.5) / (frequency_in_docs + 0.5))
            denominator = frequency + k1 * (1 - b + b * length / average_length)
            score += inverse * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum, maximum = min(values), max(values)
    if maximum == minimum:
        return [1.0 if maximum > 0 else 0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


QUESTION_STOP_WORDS = {"什么", "多少", "哪些", "如何", "怎么", "是否", "请问", "可以", "能够", "需要", "有关", "相关"}


def lexical_evidence(query: str, document: str) -> tuple[float, int]:
    """Query-term coverage before relative score normalization.

    This is a conservative lexical signal, not a calibrated probability.
    """
    words = re.findall(r"[a-zA-Z0-9_]+", query.lower())
    runs = re.findall(r"[\u4e00-\u9fff]+", query)
    bigrams = [run[index : index + 2] for run in runs for index in range(len(run) - 1)]
    terms = set(words + bigrams) - QUESTION_STOP_WORDS
    if not terms:
        return 0.0, 0
    document_tokens = set(tokenize(document))
    matched = terms & document_tokens
    return len(matched) / len(terms), len(matched)


def has_sufficient_evidence(result: dict | None, settings, semantic_provider: bool = False) -> bool:
    if not result:
        return False
    lexical = (
        result["lexical_coverage"] >= settings.min_lexical_coverage
        and result["matched_terms"] >= settings.min_matched_terms
    )
    semantic = (
        semantic_provider
        and settings.min_semantic_score > 0
        and result["dense_score"] >= settings.min_semantic_score
    )
    return lexical or semantic


class HybridRetriever:
    def __init__(self, database, embedding_provider, alpha: float = 0.65):
        self.database = database
        self.embedding_provider = embedding_provider
        self.alpha = max(0.0, min(1.0, alpha))

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        chunks = self.database.all_chunks()
        if not chunks:
            return []
        query_vector = self.embedding_provider.embed([query])[0]
        dense = [cosine_similarity(query_vector, chunk["embedding"]) for chunk in chunks]
        lexical = bm25_scores(query, [chunk["content"] for chunk in chunks])
        dense_normalized = normalize(dense)
        lexical_normalized = normalize(lexical)
        for index, chunk in enumerate(chunks):
            coverage, matched = lexical_evidence(query, chunk["content"])
            chunk["dense_score"] = dense[index]
            chunk["bm25_score"] = lexical[index]
            chunk["lexical_coverage"] = coverage
            chunk["matched_terms"] = matched
            chunk["score"] = (
                self.alpha * dense_normalized[index]
                + (1 - self.alpha) * lexical_normalized[index]
            )
        return sorted(chunks, key=lambda chunk: chunk["score"], reverse=True)[:top_k]
