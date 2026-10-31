"""Day 05: hybrid search with reciprocal rank fusion (RRF), measured with the Day 04 harness.

BM25 and word TF-IDF ranked identically on Day 04, so fusing them adds nothing. Here the second
retriever looks at character n-grams instead of words, so "PDFs" still matches "PDF" and
"tabular" shares n-grams with "tables". RRF then merges the two ranked lists by rank only.

Run:  python hybrid.py
Test: python -m pytest -q
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day-04-hit-rate-mrr"))
from evaluate import CORPUS, PARAPHRASED, QA, BM25, CosineIndex, by_sentence, evaluate, fixed_size  # noqa: E402


def char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> list[str]:
    """Character n-grams per word, padded with spaces so word starts and ends are marked."""
    grams = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        w = f" {w} "
        grams += [w[i:i + n] for n in range(n_min, n_max + 1) for i in range(len(w) - n + 1)]
    return grams


class CharNgramIndex(CosineIndex):
    """The Day 03 TF-IDF cosine index (same smoothed idf, unit rows), with character n-grams as tokens."""

    def __init__(self, docs: list[str]):
        if not docs:
            raise ValueError("need at least one document")
        grams = [char_ngrams(d) for d in docs]
        self.vocab = {g: i for i, g in enumerate(sorted({g for gs in grams for g in gs}))}
        tf = np.array([self._counts(gs) for gs in grams])
        self.idf = np.log((1 + len(docs)) / (1 + (tf > 0).sum(axis=0))) + 1
        self.matrix = self._unit(tf * self.idf)

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        scores = self.matrix @ self._unit(self._counts(char_ngrams(query)) * self.idf)
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in top]


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: each list adds 1 / (k + rank); ranks are 1-based."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


class HybridIndex:
    """Runs every retriever over all chunks, keeps only chunks with a non-zero score, fuses with RRF."""

    def __init__(self, indexes, n_docs: int, k: int = 60):
        self.indexes, self.n, self.k = indexes, n_docs, k

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        lists = [[i for i, s in idx.search(query, self.n) if s > 0] for idx in self.indexes]
        return rrf(lists, self.k)[:k]


if __name__ == "__main__":
    for name, chunks in [("fixed_size 120", fixed_size(CORPUS, 120)), ("sentence packing 160", by_sentence(CORPUS, 160))]:
        bm, ch = BM25(chunks), CharNgramIndex(chunks)
        for label, idx in [("bm25", bm), ("char-ngram", ch), ("rrf(bm25+char)", HybridIndex([bm, ch], len(chunks)))]:
            for qs_name, qs in [("original", QA), ("paraphrased", PARAPHRASED)]:
                m = evaluate(idx, chunks, qs)
                print(f"{name:<21} {label:<15} {qs_name:<12}" + "  ".join(f"{k}={v:.2f}" for k, v in m.items()))
