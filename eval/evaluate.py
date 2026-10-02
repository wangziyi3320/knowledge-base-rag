from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.database import Database
from app.embeddings import create_embedding_provider
from app.ingestion import ingest_document
from app.retrieval import HybridRetriever


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    database_path = root / "data" / "evaluation.db"
    if database_path.exists():
        database_path.unlink()
    settings = Settings(db_path=str(database_path))
    database = Database(settings.db_path)
    provider = create_embedding_provider(settings)
    sample = root / "sample_docs" / "project_handbook.md"
    ingest_document(
        database,
        provider,
        sample.name,
        "text/markdown",
        sample.read_bytes(),
        chunk_size=220,
        overlap=40,
    )
    dataset = json.loads((root / "eval" / "dataset.json").read_text(encoding="utf-8"))
    retriever = HybridRetriever(database, provider, settings.hybrid_alpha)
    hits = 0
    reciprocal_rank = 0.0
    for item in dataset:
        results = retriever.search(item["question"], top_k=5)
        rank = next(
            (
                index
                for index, result in enumerate(results, start=1)
                if any(term in result["content"] for term in item["expected_terms"])
            ),
            None,
        )
        if rank:
            hits += 1
            reciprocal_rank += 1 / rank
        print(f"question={item['question']} rank={rank} top={results[0]['content'][:50] if results else '-'}")
    count = len(dataset)
    print(json.dumps({"questions": count, "hit_at_5": hits / count, "mrr": reciprocal_rank / count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
