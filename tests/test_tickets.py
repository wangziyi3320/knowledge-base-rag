from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_ticket_draft_requires_explicit_confirmation_to_persist():
    path = Path("data/test_tickets.db")
    path.unlink(missing_ok=True)
    client = TestClient(create_app(Settings(db_path=str(path), api_key="")))

    draft_response = client.post(
        "/api/tickets/draft", json={"question": "VPN 连接失败怎么办？"}
    )
    assert draft_response.status_code == 200
    draft = draft_response.json()
    assert draft["category"] == "网络"
    assert client.get("/api/tickets").json() == []

    rejected = client.post("/api/tickets", json={**draft, "confirmed": False})
    assert rejected.status_code == 422
    assert client.get("/api/tickets").json() == []

    created = client.post("/api/tickets", json={**draft, "confirmed": True})
    assert created.status_code == 200
    assert created.json()["status"] == "open"
    ticket_id = created.json()["id"]
    assert len(client.get("/api/tickets").json()) == 1

    updated = client.patch(f"/api/tickets/{ticket_id}", json={"status": "in_progress"})
    assert updated.status_code == 200
    assert updated.json()["status"] == "in_progress"
    path.unlink(missing_ok=True)
