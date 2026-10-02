"""Reproducible, small-sample retrieval and refusal evaluation.

Run from the repository root: python -m eval.evaluate
The sample documents are fictional; these numbers are not business benchmarks.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from app.config import Settings
from app.database import Database
from app.embeddings import create_embedding_provider
from app.ingestion import ingest_document
from app.retrieval import HybridRetriever, has_sufficient_evidence


class CachedEmbeddingProvider:
    """Avoid repeating the same remote query for three ranking configurations."""

    def __init__(self, provider):
        self.provider = provider
        self.cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if missing:
            for text, vector in zip(missing, self.provider.embed(missing)):
                self.cache[text] = vector
        return [self.cache[text] for text in texts]


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percent)
    return round(ordered[index], 2)


def evidence_rank(results: list[dict], item: dict) -> int | None:
    if item["source"] is None:
        return None
    return next(
        (
            index
            for index, result in enumerate(results, start=1)
            if result["document_name"] == item["source"]
            and item["expected_quote"] in result["content"]
        ),
        None,
    )


def evaluate_config(database, provider, settings, dataset: list[dict], alpha: float) -> dict:
    retriever = HybridRetriever(database, provider, alpha=alpha)
    positives = [item for item in dataset if item["source"] is not None]
    negatives = [item for item in dataset if item["source"] is None]
    reciprocal_rank = 0.0
    hit_1 = hit_5 = accepted_positive = correctly_refused = accepted_with_evidence = 0
    latencies: list[float] = []
    cases = []
    for item in dataset:
        start = time.perf_counter()
        results = retriever.search(item["question"], top_k=5)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        rank = evidence_rank(results, item)
        accepted = has_sufficient_evidence(
            results[0] if results else None,
            settings,
            semantic_provider=settings.embedding_provider.lower() == "api",
        )
        if item["source"] is not None:
            hit_1 += rank == 1
            hit_5 += rank is not None
            reciprocal_rank += 1 / rank if rank else 0
            accepted_positive += accepted
            accepted_with_evidence += accepted and rank is not None
        else:
            correctly_refused += not accepted
        cases.append(
            {
                "question": item["question"],
                "expected_source": item["source"],
                "evidence_rank": rank,
                "would_answer": accepted,
                "top_source": results[0]["document_name"] if results else None,
                "top_lexical_coverage": round(results[0]["lexical_coverage"], 3) if results else 0,
                "latency_ms": round(elapsed_ms, 2),
            }
        )
    return {
        "alpha": alpha,
        "positive_questions": len(positives),
        "no_answer_questions": len(negatives),
        "hit_at_1": round(hit_1 / len(positives), 3),
        "hit_at_5": round(hit_5 / len(positives), 3),
        "mrr": round(reciprocal_rank / len(positives), 3),
        "positive_evidence_acceptance": round(accepted_positive / len(positives), 3),
        "accepted_with_expected_evidence_at_5": round(accepted_with_evidence / len(positives), 3),
        "no_answer_rejection": round(correctly_refused / len(negatives), 3),
        "latency_ms_p50": percentile(latencies, 0.5),
        "latency_ms_p95": percentile(latencies, 0.95),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--details", action="store_true", help="Print per-question records")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    (root / "data").mkdir(exist_ok=True)
    dataset = json.loads((root / "eval" / "dataset.json").read_text(encoding="utf-8"))
    settings = Settings()
    evaluation_path = root / "data" / f"evaluation-{uuid.uuid4().hex}.db"
    try:
        database = Database(str(evaluation_path))
        provider = CachedEmbeddingProvider(create_embedding_provider(settings))
        document_names = sorted({item["source"] for item in dataset if item["source"] is not None})
        for name in document_names:
            document = root / "sample_docs" / name
            ingest_document(
                database, provider, name, "text/markdown", document.read_bytes(),
                chunk_size=220, overlap=40,
            )
        results = {
            "note": "Synthetic sample only; retrieval plus evidence-gate metrics, not generated-answer quality",
            "provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model if settings.embedding_provider == "api" else "hash-demo",
            "document_count": len(document_names),
            "question_count": len(dataset),
            "thresholds": {
                "min_lexical_coverage": settings.min_lexical_coverage,
                "min_matched_terms": settings.min_matched_terms,
                "min_semantic_score": settings.min_semantic_score,
            },
            "configurations": {
                "bm25": evaluate_config(database, provider, settings, dataset, 0.0),
                "hybrid": evaluate_config(database, provider, settings, dataset, settings.hybrid_alpha),
                "dense": evaluate_config(database, provider, settings, dataset, 1.0),
            },
        }
    finally:
        evaluation_path.unlink(missing_ok=True)
    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    summary = dict(results)
    summary["configurations"] = {
        name: {key: value for key, value in value.items() if key != "cases"}
        for name, value in results["configurations"].items()
    }
    print(rendered if args.details else json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
