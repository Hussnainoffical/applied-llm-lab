"""Day 10: cross-encoder reranking on top of the fused retriever (RAG v2, step 2).

Day 09 fused BM25, character n-grams and bge-small with RRF. A bi-encoder scores query and chunk
separately; a cross-encoder reads them together, which is slower but sees word interactions.
Here Xenova/ms-marco-MiniLM-L-6-v2 (ONNX via fastembed, CPU) reorders the first-stage top `pool`
chunks, measured with the unchanged Day 04 harness and timed per query.

Run:  python rerank.py   (downloads the ONNX models on first use)
Test: python -m pytest -q
"""
import math
import sys
import time
from pathlib import Path
from typing import Callable

for day in ("day-04-hit-rate-mrr", "day-05-hybrid-rrf", "day-09-dense-vs-sparse"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / day))
from evaluate import evaluate  # noqa: E402
from dense import CORPUS, PARAPHRASED, QA, BM25, CharNgramIndex, DenseIndex, HybridIndex, bge_small, by_sentence  # noqa: E402

Score = Callable[[str, list[str]], list[float]]


class RerankIndex:
    """First stage proposes `pool` chunks; the cross-encoder reorders them; the rest keep their order."""

    def __init__(self, first, chunks: list[str], score: Score, pool: int = 5):
        if pool < 1:
            raise ValueError("pool must be >= 1")
        self.first, self.chunks, self.score, self.pool = first, chunks, score, pool
        self.seconds = 0.0  # time spent in the cross-encoder, for the latency report

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        hits = self.first.search(query, max(k, self.pool))
        head, tail = hits[: self.pool], hits[self.pool:]
        t = time.perf_counter()
        scores = self.score(query, [self.chunks[i] for i, _ in head])
        self.seconds += time.perf_counter() - t
        # logits can be negative; the Day 04 harness drops scores <= 0, so map them to (0, 1)
        head = sorted(((i, 1 / (1 + math.exp(-float(s)))) for (i, _), s in zip(head, scores)), key=lambda x: -x[1])
        return (head + tail)[:k]


def minilm() -> Score:
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    model = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    return lambda q, docs: list(model.rerank(q, docs))


if __name__ == "__main__":
    chunks = by_sentence(CORPUS, 160)
    docs, query = bge_small()
    fused = HybridIndex([BM25(chunks), CharNgramIndex(chunks), DenseIndex(chunks, docs, query)], len(chunks))
    score = minilm()
    print(f"sentence packing 160: {len(chunks)} chunks")
    for label, idx in [("rrf(bm25+char+dense)", fused)] + [(f"+ rerank pool={p}", RerankIndex(fused, chunks, score, p)) for p in (3, 5, len(chunks))]:
        for qs, q in (("original", QA), ("paraphrased", PARAPHRASED)):
            m = evaluate(idx, chunks, q)
            ms = 1000 * getattr(idx, "seconds", 0.0) / len(q)
            if hasattr(idx, "seconds"):
                idx.seconds = 0.0
            print(f"  {label:<22} {qs:<12} hit@1={m['hit@1']:.2f} hit@3={m['hit@3']:.2f} mrr@5={m['mrr@5']:.2f}  rerank {ms:.1f} ms/query")
