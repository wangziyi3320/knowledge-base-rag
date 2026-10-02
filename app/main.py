from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings
from app.database import Database
from app.embeddings import create_embedding_provider
from app.generation import AnswerGenerator
from app.ingestion import ingest_document
from app.retrieval import HybridRetriever, has_sufficient_evidence
from app.schemas import (
    Citation, DocumentInfo, QueryRequest, QueryResponse, TicketCreate, TicketDraft,
    TicketDraftRequest, TicketInfo, TicketStatusUpdate,
)
from app.tickets import make_ticket_draft


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    database = Database(settings.db_path)
    embedding_provider = create_embedding_provider(settings)
    database.ensure_embedding_profile(embedding_provider.profile)
    retriever = HybridRetriever(database, embedding_provider, settings.hybrid_alpha)
    generator = AnswerGenerator(settings)

    application = FastAPI(title="Knowledge Base RAG", version="0.1.0")
    static_dir = Path(__file__).parent / "static"
    application.mount("/static", StaticFiles(directory=static_dir), name="static")

    @application.get("/", include_in_schema=False)
    def index():
        return FileResponse(static_dir / "index.html")

    @application.get("/health")
    def health():
        return {"status": "ok", "embedding_provider": settings.embedding_provider}

    @application.get("/api/documents", response_model=list[DocumentInfo])
    def list_documents():
        return database.list_documents()

    @application.post("/api/documents")
    async def upload_document(file: UploadFile = File(...)):
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit")
        try:
            return ingest_document(
                database,
                embedding_provider,
                file.filename or "document.txt",
                file.content_type or "application/octet-stream",
                content,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail=f"Embedding API request failed: {error}") from error

    @application.delete("/api/documents/{document_id}")
    def delete_document(document_id: int):
        if not database.delete_document(document_id):
            raise HTTPException(status_code=404, detail="Document not found")
        return {"deleted": True}

    @application.post("/api/query", response_model=QueryResponse)
    def query(request: QueryRequest):
        try:
            results = retriever.search(request.question, request.top_k)
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail=f"Embedding API request failed: {error}") from error
        answerable = has_sufficient_evidence(
            results[0] if results else None,
            settings,
            semantic_provider=settings.embedding_provider.lower() == "api",
        )
        if answerable:
            try:
                answer, mode = generator.generate(request.question, results)
            except httpx.HTTPError as error:
                raise HTTPException(status_code=502, detail=f"Model API request failed: {error}") from error
            if mode in {"unverified-generation", "insufficient-evidence"}:
                answerable = False
        else:
            answer, mode = "知识库中没有足够信息。", "insufficient-evidence"
        citations = [
            Citation(
                rank=index,
                document_id=item["document_id"],
                document_name=item["document_name"],
                chunk_id=item["id"],
                content=item["content"],
                score=round(float(item["score"]), 4),
            )
            for index, item in enumerate(results, start=1)
        ]
        if answerable:
            reason = "通过最低证据条件；请继续核对原文。"
        elif mode == "unverified-generation":
            reason = "模型回答未提供有效引用，已拒绝该回答。"
        else:
            reason = "当前知识片段不足以支持回答。"
        return QueryResponse(
            answer=answer,
            citations=citations,
            generation_mode=mode,
            answerable=answerable,
            evidence_reason=reason,
        )

    @application.post("/api/tickets/draft", response_model=TicketDraft)
    def draft_ticket(request: TicketDraftRequest):
        return make_ticket_draft(request.question)

    @application.post("/api/tickets", response_model=TicketInfo)
    def create_ticket(request: TicketCreate):
        # Pydantic requires confirmed=true; the draft endpoint never persists data.
        return database.create_ticket(request.title, request.description, request.category)

    @application.get("/api/tickets", response_model=list[TicketInfo])
    def list_tickets():
        return database.list_tickets()

    @application.patch("/api/tickets/{ticket_id}", response_model=TicketInfo)
    def update_ticket_status(ticket_id: int, request: TicketStatusUpdate):
        ticket = database.update_ticket_status(ticket_id, request.status)
        if ticket is None:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return ticket

    return application


app = create_app()
