from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[。！？!?；;])|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100")
    sentences = split_sentences(text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            step = max(1, chunk_size - overlap)
            for start in range(0, len(sentence), step):
                piece = sentence[start : start + chunk_size]
                if piece:
                    chunks.append(piece)
            continue
        candidate = sentence if not current else f"{current}\n{sentence}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n{sentence}".strip()
    if current:
        chunks.append(current)
    return chunks
