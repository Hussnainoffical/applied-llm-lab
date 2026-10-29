"""Day 03: cosine search over NumPy TF-IDF vectors, compared with Day 02 BM25.

Run:  python cosine.py
Test: python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day-02-bm25"))
from bm25 import BM25, QA, SAMPLE, by_sentence, fixed_size, tokenize  # noqa: E402


class CosineIndex:
    """TF-IDF rows, L2-normalised, so a search is one matrix-vector product."""

    def __init__(self, docs: list[str]):
        if not docs:
            raise ValueError("need at least one document")
        toks = [tokenize(d) for d in docs]
        self.vocab = {t: i for i, t in enumerate(sorted({t for ts in toks for t in ts}))}
        tf = np.array([self._counts(ts) for ts in toks])
        df = (tf > 0).sum(axis=0)
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1  # smoothed idf, never zero
        self.matrix = self._unit(tf * self.idf)

    def _counts(self, tokens: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t in tokens:
            if t in self.vocab:
                v[self.vocab[t]] += 1
        return v

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return np.divide(x, n, out=np.zeros_like(x), where=n > 0)

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        scores = self.matrix @ self._unit(self._counts(tokenize(query)) * self.idf)
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in top]


# Same facts as Day 02, asked with different words than the text uses.
PARAPHRASES = [
    ("why can cutting text into pieces hurt answers", "a fact split across two chunks may never be retrieved whole"),
    ("what happens to every piece before storage", "Each chunk is embedded and stored"),
    ("which pieces match exactly but miss surrounding meaning", "Smaller chunks give precise matches but lose context"),
]


def top1_hits(index, chunks: list[str], qa) -> int:
    """A hit needs a non-zero score and the whole fact inside the top chunk."""
    flat = [" ".join(c.split()) for c in chunks]
    hits = 0
    for q, fact in qa:
        i, score = index.search(q, 1)[0]
        hits += score > 0 and fact in flat[i]
    return hits


if __name__ == "__main__":
    for name, chunks in [("fixed_size 120", fixed_size(SAMPLE, 120)),
                         ("fixed 120 + 30 overlap", fixed_size(SAMPLE, 120, 30)),
                         ("sentence packing 160", by_sentence(SAMPLE, 160))]:
        bm, cos = BM25(chunks), CosineIndex(chunks)
        print(f"{name:<24} original Qs  bm25={top1_hits(bm, chunks, QA)}/{len(QA)}  "
              f"cosine={top1_hits(cos, chunks, QA)}/{len(QA)}  |  paraphrased Qs  "
              f"bm25={top1_hits(bm, chunks, PARAPHRASES)}/{len(PARAPHRASES)}  "
              f"cosine={top1_hits(cos, chunks, PARAPHRASES)}/{len(PARAPHRASES)}")
