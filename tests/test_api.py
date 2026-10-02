from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
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

    deletion = client.delete(f"/api/documents/{document_id}")
    assert deletion.status_code == 200
    assert client.get("/api/documents").json() == []
    path.unlink(missing_ok=True)
