from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    id: int
    name: str
    content_type: str
    chunk_count: int
    created_at: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=12)


class Citation(BaseModel):
    rank: int
    document_id: int
    document_name: str
    chunk_id: int
    content: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    generation_mode: str

