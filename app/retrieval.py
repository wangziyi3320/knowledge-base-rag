from __future__ import annotations

import math
from collections import Counter

from app.embeddings import tokenize


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


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
            chunk["score"] = (
                self.alpha * dense_normalized[index]
                + (1 - self.alpha) * lexical_normalized[index]
            )
        return sorted(chunks, key=lambda chunk: chunk["score"], reverse=True)[:top_k]

