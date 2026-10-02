from app.chunking import chunk_text, normalize_text


def test_normalize_text_removes_duplicate_whitespace():
    assert normalize_text("第一段。  \n\n\n第二段。") == "第一段。\n\n第二段。"


def test_chunk_text_keeps_content_and_creates_multiple_chunks():
    text = "。".join([f"第{i}条项目说明包含一些用于测试的内容" for i in range(30)])
    chunks = chunk_text(text, chunk_size=120, overlap=20)
    assert len(chunks) > 2
    assert "第0条" in chunks[0]
    assert any("第29条" in chunk for chunk in chunks)

