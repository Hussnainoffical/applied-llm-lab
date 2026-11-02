"""Day 07: Recall@k curves - how deep must the candidate pool be before a reranker?

A reranker can only reorder what the first stage hands it. If the chunk holding the whole fact is
not in the candidate pool, no reranker can bring it back. So before adding one, measure Recall@k
for k = 1..N: the share of questions whose whole-fact chunk is somewhere in the top k, and the
smallest pool depth at which recall stops growing.

Run:  python recall_depth.py
Test: python -m pytest -q
"""
import sys
from pathlib import Path

for day in ("day-04-hit-rate-mrr", "day-05-hybrid-rrf"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / day))
from evaluate import answer_rank  # noqa: E402
from hybrid import CORPUS, PARAPHRASED, QA, BM25, CharNgramIndex, HybridIndex, by_sentence, fixed_size  # noqa: E402


def recall_curve(index, chunks: list[str], qa, max_k: int) -> list[float]:
    """recall[k-1] = share of questions whose whole-fact chunk is ranked within the top k."""
    ranks = [answer_rank(index, chunks, q, f, max_k) for q, f in qa]
    return [sum(r is not None and r <= k for r in ranks) / len(qa) for k in range(1, max_k + 1)]


def ceiling(chunks: list[str], qa) -> float:
    """Best recall any retriever could reach: share of facts that sit whole in at least one chunk."""
    flat = [" ".join(c.split()) for c in chunks]
    return sum(any(f in c for c in flat) for _, f in qa) / len(qa)


def saturation_depth(curve: list[float]) -> int:
    """Smallest k at which the curve reaches its final value."""
    return next(k for k, r in enumerate(curve, 1) if r == curve[-1])


if __name__ == "__main__":
    for name, chunks in [("fixed_size 120", fixed_size(CORPUS, 120)), ("sentence packing 160", by_sentence(CORPUS, 160))]:
        bm, ch = BM25(chunks), CharNgramIndex(chunks)
        print(f"{name}: {len(chunks)} chunks, recall ceiling {ceiling(chunks, QA):.2f}")
        for label, idx in [("bm25", bm), ("char-ngram", ch), ("rrf(bm25+char)", HybridIndex([bm, ch], len(chunks)))]:
            for qs_name, qs in [("original", QA), ("paraphrased", PARAPHRASED)]:
                c = recall_curve(idx, chunks, qs, len(chunks))
                print(f"  {label:<15} {qs_name:<12} R@1={c[0]:.2f} R@3={c[2]:.2f} R@5={c[4]:.2f} "
                      f"R@{len(chunks)}={c[-1]:.2f}  saturates at k={saturation_depth(c)}")
