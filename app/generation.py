from __future__ import annotations

import re

import httpx

from app.retrieval import lexical_evidence


SYSTEM_PROMPT = """你是企业知识库问答助手。只能依据提供的资料回答。
每个事实后使用 [数字] 标注引用。资料不足时明确回答“知识库中没有足够信息”。
不要编造来源、数字或结论。"""


def relevant_excerpt(question: str, content: str, limit: int = 220) -> str:
    """在纯检索模式显示与问题词面最接近的句子，避免只截取长片段开头。"""
    sentences = [part.strip() for part in re.split(r"(?<=[。！？；])|\n+", content) if part.strip()]
    if not sentences:
        return content[:limit]
    best = max(sentences, key=lambda part: (*lexical_evidence(question, part), -len(part)))
    return best[:limit]


class AnswerGenerator:
    def __init__(self, settings):
        self.settings = settings

    def generate(self, question: str, results: list[dict]) -> tuple[str, str]:
        if not results:
            return "知识库中没有足够信息。请先上传相关文档。", "no-context"
        if not self.settings.api_key:
            snippet = relevant_excerpt(question, results[0]["content"])
            return f"未配置大模型 API，以下为最相关的检索片段：\n\n[1] {snippet}", "retrieval-only"
        context = "\n\n".join(
            f"[{index}] 来源：{item['document_name']}\n{item['content']}"
            for index, item in enumerate(results, start=1)
        )
        response = httpx.post(
            self.settings.api_base_url.rstrip("/") + "/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.settings.api_key}"},
            json={
                "model": self.settings.chat_model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
                ],
            },
            timeout=90,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
        if "知识库中没有足够信息" in answer:
            return "知识库中没有足够信息。", "insufficient-evidence"
        references = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
        if not references or any(value < 1 or value > len(results) for value in references):
            return "知识库中没有足够信息。模型回答缺少可验证的引用。", "unverified-generation"
        return answer, "llm"
