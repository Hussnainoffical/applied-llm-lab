"""Day 02: BM25 retrieval from scratch, used to rank the Day 01 chunks.

Run:  python bm25.py
Test: python -m pytest -q
"""
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day-01-chunking"))
from chunking import by_sentence, fixed_size  # noqa: E402

STOP = {"a", "an", "the", "is", "are", "of", "to", "and", "in", "it", "at", "but", "be", "what", "how", "do", "does", "why"}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


class BM25:
    """Okapi BM25: term frequency saturates (k1) and long chunks are penalised (b)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        if not docs:
            raise ValueError("need at least one document")
        self.docs, self.k1, self.b = docs, k1, b
        toks = [tokenize(d) for d in docs]
        self.tf = [Counter(t) for t in toks]
        self.len = [len(t) for t in toks]
        self.avgdl = sum(self.len) / len(docs) or 1
        df = Counter(term for t in toks for term in set(t))
        n = len(docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def score(self, query: str, i: int) -> float:
        norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl)
        return sum(self.idf[t] * f * (self.k1 + 1) / (f + norm)
                   for t in tokenize(query) if (f := self.tf[i].get(t, 0)))

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        ranked = sorted(((i, self.score(query, i)) for i in range(len(self.docs))), key=lambda x: -x[1])
        return ranked[:k]


SAMPLE = (
    "Retrieval-augmented generation grounds an LLM in your own documents. "
    "First the documents are split into chunks. Each chunk is embedded and stored.\n\n"
    "At question time, the most similar chunks are retrieved. They are passed to the model "
    "as context. Chunk boundaries matter: a fact split across two chunks may never be retrieved whole.\n\n"
    "Smaller chunks give precise matches but lose context. Larger chunks keep context but add noise."
)
# (question, the fact the top chunk must contain in full)
QA = [
    ("why do chunk boundaries matter", "a fact split across two chunks may never be retrieved whole"),
    ("what is done with each chunk", "Each chunk is embedded and stored"),
    ("what do smaller chunks give", "Smaller chunks give precise matches but lose context"),
    ("what does retrieval-augmented generation ground", "grounds an LLM in your own documents"),
    ("what is passed to the model as context", "They are passed to the model as context"),
]


def whole_fact_at_top1(chunks: list[str]) -> int:
    bm = BM25(chunks)
    flat = [" ".join(c.split()) for c in chunks]
    return sum(fact in flat[bm.search(q, 1)[0][0]] for q, fact in QA)


if __name__ == "__main__":
    for name, chunks in [("fixed_size 120", fixed_size(SAMPLE, 120)),
                         ("fixed 120 + 30 overlap", fixed_size(SAMPLE, 120, 30)),
                         ("sentence packing 160", by_sentence(SAMPLE, 160))]:
        print(f"{name:<24} chunks={len(chunks)}  whole fact in top-1: {whole_fact_at_top1(chunks)}/{len(QA)}")
