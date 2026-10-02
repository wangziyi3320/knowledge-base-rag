from pathlib import Path

from app.database import Database
from app.embeddings import HashEmbeddingProvider
from app.retrieval import HybridRetriever, cosine_similarity, has_sufficient_evidence
from app.config import Settings


def test_hybrid_retrieval_returns_relevant_chunk():
    path = Path("data/test_retrieval.db")
    path.unlink(missing_ok=True)
    database = Database(str(path))
    provider = HashEmbeddingProvider(128)
    document_id = database.create_document("handbook.md", "text/markdown", "checksum")
    texts = [
        "发布完成后需要执行冒烟测试，并记录回滚方案。",
        "每次迭代结束后需要进行项目复盘。",
        "开发人员从主分支创建功能分支。",
    ]
    vectors = provider.embed(texts)
    database.add_chunks(document_id, [(i, text, vectors[i]) for i, text in enumerate(texts)])
    results = HybridRetriever(database, provider).search("发布后要做什么测试", top_k=2)
    assert results
    assert "冒烟测试" in results[0]["content"]
    path.unlink(missing_ok=True)


def test_cosine_uses_vector_norms_and_handles_zero():
    assert cosine_similarity([2.0, 0.0], [5.0, 0.0]) == 1.0
    assert cosine_similarity([0.0, 0.0], [5.0, 0.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_evidence_gate_does_not_trust_relative_ranking_score():
    settings = Settings()
    weak_top_result = {
        "score": 1.0,
        "lexical_coverage": 0.0,
        "matched_terms": 0,
        "dense_score": 0.95,
    }
    assert not has_sufficient_evidence(weak_top_result, settings, semantic_provider=False)
