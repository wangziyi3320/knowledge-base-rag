from __future__ import annotations

from typing import Literal

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
    answerable: bool
    evidence_reason: str


TicketCategory = Literal["账号", "网络", "设备", "财务", "其他"]


class TicketDraftRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


class TicketDraft(BaseModel):
    title: str = Field(min_length=2, max_length=100)
    description: str = Field(min_length=2, max_length=2000)
    category: TicketCategory


class TicketCreate(TicketDraft):
    confirmed: Literal[True]


class TicketInfo(TicketDraft):
    id: int
    status: Literal["open", "in_progress", "resolved"]
    created_at: str


class TicketStatusUpdate(BaseModel):
    status: Literal["open", "in_progress", "resolved"]
