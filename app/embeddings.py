from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx


def tokenize(text: str) -> list[str]:
    latin = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    bigrams = ["".join(chinese[i : i + 2]) for i in range(len(chinese) - 1)]
    return latin + chinese + bigrams


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbeddingProvider:
    """Deterministic local embedding for offline demos and automated tests."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimension
            sign = 1.0 if (value >> 1) % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OpenAICompatibleEmbeddingProvider:
    def __init__(self, base_url: str, api_key: str, model: str):
        if not api_key or not model:
            raise ValueError("API key and embedding model are required")
        self.url = base_url.rstrip("/") + "/v1/embeddings"
        self.api_key = api_key
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in data]


def create_embedding_provider(settings) -> EmbeddingProvider:
    if settings.embedding_provider.lower() == "api":
        return OpenAICompatibleEmbeddingProvider(
            settings.api_base_url, settings.api_key, settings.embedding_model
        )
    return HashEmbeddingProvider(settings.embedding_dimension)

