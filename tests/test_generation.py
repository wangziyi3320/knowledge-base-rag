import httpx

from app.config import Settings
from app.generation import AnswerGenerator, relevant_excerpt


def test_retrieval_only_excerpt_uses_relevant_sentence():
    content = "账号开通后会收到通知。\nVPN 连接失败时先检查客户端版本与网络状态，再提交服务台工单。"
    excerpt = relevant_excerpt("VPN 连接失败时怎么办？", content)
    assert "连接失败时先检查" in excerpt
    assert "账号开通" not in excerpt


def test_retrieval_only_answer_uses_top_candidate_without_unrelated_followup():
    answer, mode = AnswerGenerator(Settings(api_key="")).generate(
        "VPN 连接失败怎么办？",
        [
            {"content": "VPN 连接失败时先检查客户端版本。"},
            {"content": "电脑无法启动时检查电源连接。"},
        ],
    )
    assert mode == "retrieval-only"
    assert "先检查客户端版本" in answer
    assert "检查电源连接" not in answer


def test_llm_answer_without_valid_citation_is_not_accepted(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "发布后执行冒烟测试。[9]"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    generator = AnswerGenerator(Settings(api_key="test-token"))
    answer, mode = generator.generate(
        "发布后怎么办？",
        [{"document_name": "guide.md", "content": "发布后执行冒烟测试。"}],
    )
    assert mode == "unverified-generation"
    assert "缺少可验证的引用" in answer
