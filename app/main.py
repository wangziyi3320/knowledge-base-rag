from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings
from app.database import Database
from app.embeddings import create_embedding_provider
from app.generation import AnswerGenerator
from app.ingestion import ingest_document
from app.retrieval import HybridRetriever
from app.schemas import Citation, DocumentInfo, QueryRequest, QueryResponse


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    database = Database(settings.db_path)
    embedding_provider = create_embedding_provider(settings)
    retriever = HybridRetriever(database, embedding_provider, settings.hybrid_alpha)
    generator = AnswerGenerator(settings)

    application = FastAPI(title="Knowledge Base RAG", version="0.1.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    @application.delete("/api/documents/{document_id}")
    def delete_document(document_id: int):
        if not database.delete_document(document_id):
            raise HTTPException(status_code=404, detail="Document not found")
        return {"deleted": True}

    @application.post("/api/query", response_model=QueryResponse)
    def query(request: QueryRequest):
        results = retriever.search(request.question, request.top_k)
        try:
            answer, mode = generator.generate(request.question, results)
        except httpx.HTTPError as error:  # type: ignore[name-defined]
            raise HTTPException(status_code=502, detail=f"Model API request failed: {error}") from error
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
        return QueryResponse(answer=answer, citations=citations, generation_mode=mode)

    return application


import httpx  # imported late to keep endpoint error handling explicit

app = create_app()

