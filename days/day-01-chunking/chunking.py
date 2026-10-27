"""Day 01: three ways to chunk text for RAG, and how they differ.

Run:  python chunking.py
Test: python -m pytest -q
"""
import re


def fixed_size(text: str, size: int = 200, overlap: int = 0) -> list[str]:
    """Split by character count. Simple and predictable, but cuts mid-sentence."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def by_sentence(text: str, max_chars: int = 200) -> list[str]:
    """Pack whole sentences into chunks up to max_chars. Keeps meaning intact."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks, current = [], ""
    for s in sentences:
        if current and len(current) + 1 + len(s) > max_chars:
            chunks.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        chunks.append(current)
    return chunks


def by_paragraph(text: str) -> list[str]:
    """Split on blank lines. Best when the document already has structure."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def report(name: str, chunks: list[str]) -> str:
    sizes = [len(c) for c in chunks]
    cut = sum(1 for c in chunks if not c.rstrip().endswith((".", "!", "?")))
    return f"{name:<14} chunks={len(chunks):<3} avg={sum(sizes) // len(sizes):<4} cut_mid_sentence={cut}"


if __name__ == "__main__":
    sample = (
        "Retrieval-augmented generation grounds an LLM in your own documents. "
        "First the documents are split into chunks. Each chunk is embedded and stored.\n\n"
        "At question time, the most similar chunks are retrieved. They are passed to the model "
        "as context. Chunk boundaries matter: a fact split across two chunks may never be retrieved whole.\n\n"
        "Smaller chunks give precise matches but lose context. Larger chunks keep context but add noise."
    )
    print(report("fixed_size", fixed_size(sample, 120)))
    print(report("fixed+overlap", fixed_size(sample, 120, 30)))
    print(report("by_sentence", by_sentence(sample, 160)))
    print(report("by_paragraph", by_paragraph(sample)))
