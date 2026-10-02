from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from app.config import Settings
from app.database import Database
from app.embeddings import OpenAICompatibleEmbeddingProvider, create_embedding_provider
from app.main import create_app


def test_upload_query_and_delete():
    path = Path("data/test_api.db")
    path.unlink(missing_ok=True)
    settings = Settings(db_path=str(path), api_key="")
    client = TestClient(create_app(settings))

    upload = client.post(
        "/api/documents",
        files={"file": ("guide.md", "发布后需要执行冒烟测试并准备回滚方案。", "text/markdown")},
    )
    assert upload.status_code == 200
    document_id = upload.json()["document_id"]

    query = client.post("/api/query", json={"question": "发布后需要做什么？", "top_k": 3})
    assert query.status_code == 200
    assert query.json()["citations"][0]["document_name"] == "guide.md"
    assert query.json()["generation_mode"] == "retrieval-only"
    assert query.json()["answerable"] is True

    unknown = client.post("/api/query", json={"question": "员工婚假有多少天？"})
    assert unknown.status_code == 200
    assert unknown.json()["answerable"] is False
    assert unknown.json()["generation_mode"] == "insufficient-evidence"

    deletion = client.delete(f"/api/documents/{document_id}")
    assert deletion.status_code == 200
    assert client.get("/api/documents").json() == []
    path.unlink(missing_ok=True)


def test_embedding_service_failure_returns_502(monkeypatch):
    path = Path("data/test_api_remote.db")
    path.unlink(missing_ok=True)
    settings = Settings(
        db_path=str(path), embedding_provider="api",
        embedding_api_base_url="https://embed.example.test",
        embedding_api_key="test-token", embedding_model="chinese-model",
    )
    database = Database(str(path))
    database.ensure_embedding_profile(create_embedding_provider(settings).profile)
    document_id = database.create_document("guide.md", "text/markdown", "checksum")
    database.add_chunks(document_id, [(0, "发布后执行冒烟测试。", [1.0, 0.0])])
    client = TestClient(create_app(settings))

    def fail_embed(self, texts):
        raise httpx.ConnectError("service unavailable")

    monkeypatch.setattr(OpenAICompatibleEmbeddingProvider, "embed", fail_embed)
    assert client.post("/api/query", json={"question": "发布后做什么测试？"}).status_code == 502
    assert client.post(
        "/api/documents",
        files={"file": ("new.md", "新的测试文档。", "text/markdown")},
    ).status_code == 502
    path.unlink(missing_ok=True)
