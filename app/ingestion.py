from __future__ import annotations

import hashlib
import io
from pathlib import Path

from pypdf import PdfReader

from app.chunking import chunk_text


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Only TXT, Markdown and PDF files are supported")
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return content.decode("utf-8-sig").strip()


def ingest_document(
    database,
    embedding_provider,
    filename: str,
    content_type: str,
    content: bytes,
    chunk_size: int = 500,
    overlap: int = 80,
) -> dict:
    checksum = hashlib.sha256(content).hexdigest()
    existing = database.find_document_by_checksum(checksum)
    if existing:
        return {"document_id": existing["id"], "chunk_count": 0, "duplicate": True}
    text = extract_text(filename, content)
    if not text:
        raise ValueError("No readable text was found in the document")
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    embeddings = embedding_provider.embed(chunks)
    document_id = database.create_document(filename, content_type, checksum)
    database.add_chunks(
        document_id,
        [(index, chunk, embeddings[index]) for index, chunk in enumerate(chunks)],
    )
    return {"document_id": document_id, "chunk_count": len(chunks), "duplicate": False}
