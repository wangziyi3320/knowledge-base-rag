from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    db_path: str = os.getenv("RAG_DB_PATH", "data/knowledge_base.db")
    embedding_provider: str = os.getenv("RAG_EMBEDDING_PROVIDER", "hash")
    embedding_dimension: int = int(os.getenv("RAG_EMBEDDING_DIMENSION", "384"))
    api_base_url: str = os.getenv("RAG_API_BASE_URL", "https://api.deepseek.com")
    api_key: str = os.getenv("RAG_API_KEY", "")
    chat_model: str = os.getenv("RAG_CHAT_MODEL", "deepseek-chat")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "")
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    hybrid_alpha: float = float(os.getenv("RAG_HYBRID_ALPHA", "0.65"))

