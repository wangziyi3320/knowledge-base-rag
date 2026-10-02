from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable


class Database:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def init_schema(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    checksum TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
                """
            )

    def create_document(self, name: str, content_type: str, checksum: str) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO documents(name, content_type, checksum) VALUES (?, ?, ?)",
                (name, content_type, checksum),
            )
            return int(cursor.lastrowid)

    def find_document_by_checksum(self, checksum: str):
        with self.connect() as connection:
            return connection.execute(
                "SELECT id, name FROM documents WHERE checksum = ?", (checksum,)
            ).fetchone()

    def add_chunks(self, document_id: int, chunks: Iterable[tuple[int, str, list[float]]]) -> None:
        rows = [
            (document_id, position, content, json.dumps(embedding))
            for position, content, embedding in chunks
        ]
        with self.connect() as connection:
            connection.executemany(
                "INSERT INTO chunks(document_id, position, content, embedding) VALUES (?, ?, ?, ?)",
                rows,
            )

    def list_documents(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT d.id, d.name, d.content_type, d.created_at, COUNT(c.id) AS chunk_count
                FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id ORDER BY d.created_at DESC, d.id DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_document(self, document_id: int) -> bool:
        with self.connect() as connection:
            cursor = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            return cursor.rowcount > 0

    def all_chunks(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT c.id, c.document_id, c.position, c.content, c.embedding, d.name AS document_name
                FROM chunks c JOIN documents d ON d.id = c.document_id
                ORDER BY c.document_id, c.position
                """
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["embedding"] = json.loads(item["embedding"])
                result.append(item)
            return result
