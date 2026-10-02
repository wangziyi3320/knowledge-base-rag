from __future__ import annotations

import httpx


SYSTEM_PROMPT = """你是企业知识库问答助手。只能依据提供的资料回答。
每个事实后使用 [数字] 标注引用。资料不足时明确回答“知识库中没有足够信息”。
不要编造来源、数字或结论。"""


class AnswerGenerator:
    def __init__(self, settings):
        self.settings = settings

    def generate(self, question: str, results: list[dict]) -> tuple[str, str]:
        if not results:
            return "知识库中没有足够信息。请先上传相关文档。", "no-context"
        if not self.settings.api_key:
            lines = ["未配置大模型 API，以下为检索到的相关资料："]
            for index, result in enumerate(results, start=1):
                snippet = result["content"].replace("\n", " ")[:220]
                lines.append(f"[{index}] {snippet}")
            return "\n\n".join(lines), "retrieval-only"
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
        return answer, "llm"

