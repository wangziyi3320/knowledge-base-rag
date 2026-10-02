import httpx
import pytest
from pathlib import Path

from app.config import Settings
from app.database import Database
from app.embeddings import OpenAICompatibleEmbeddingProvider, create_embedding_provider


def test_openai_compatible_embedding_orders_response_and_uses_separate_credentials(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, payload=json)
        return httpx.Response(
            200,
            json={"data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [2.0, 0.0]},
            ]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    settings = Settings(
        embedding_provider="api", embedding_api_base_url="https://embed.example.test",
        embedding_api_key="test-token", embedding_model="chinese-model",
        api_base_url="https://chat.example.test", api_key="chat-token",
    )
    provider = create_embedding_provider(settings)
    assert isinstance(provider, OpenAICompatibleEmbeddingProvider)
    assert provider.embed(["第一段", "第二段"]) == [[2.0, 0.0], [0.0, 1.0]]
    assert captured["url"] == "https://embed.example.test/v1/embeddings"
    assert captured["headers"]["Authorization"] == "Bearer test-token"
    assert captured["payload"]["model"] == "chinese-model"


def test_unknown_embedding_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        create_embedding_provider(Settings(embedding_provider="typo"))


def test_existing_index_rejects_embedding_model_switch():
    path = Path("data/test_profile.db")
    path.unlink(missing_ok=True)
    database = Database(str(path))
    database.ensure_embedding_profile("hash:384")
    database.create_document("guide.md", "text/markdown", "checksum")
    with pytest.raises(ValueError, match="Use a new RAG_DB_PATH"):
        database.ensure_embedding_profile("api:https://other.test/v1/embeddings:chinese-model")
    path.unlink(missing_ok=True)
