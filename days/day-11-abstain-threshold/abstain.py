"""Day 11: when should retrieval say "not found"? (RAG v2, step 3)

Day 10 reranked the fused top 5 with a MiniLM cross-encoder. Here 12 on-topic questions that the
corpus cannot answer are added. For every question only the top-1 hit and its score are kept, and two
signals are compared as an abstention rule: the RRF score of the fused list and the cross-encoder
probability after reranking. AUROC says how well a signal separates answerable from unanswerable
questions; the threshold sweep shows what a fixed cut-off would do to both groups.

Run:  python abstain.py   (uses the Day 09/10 ONNX models, CPU)
Test: python -m pytest -q
"""
import sys
from pathlib import Path

for day in ("day-04-hit-rate-mrr", "day-05-hybrid-rrf", "day-09-dense-vs-sparse", "day-10-cross-encoder-rerank"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / day))
from rerank import (CORPUS, PARAPHRASED, QA, BM25, CharNgramIndex, DenseIndex, HybridIndex,  # noqa: E402
                    RerankIndex, bge_small, by_sentence, minilm)

# on-topic for a RAG corpus, but the corpus never states the answer
UNANSWERABLE = [
    "which GPU is needed to train the embedding model",
    "how much does a vector database cost per month",
    "which language model writes the final answer",
    "how long does indexing take for a thousand PDFs",
    "who invented reciprocal rank fusion",
    "how many milliseconds does the reranker need on a CPU",
    "which license does FAISS use",
    "how many tokens fit into the context window",
    "can BM25 search inside images",
    "what learning rate is used to fine-tune the reranker",
    "how are chunks translated into German",
    "how often should the index be rebuilt",
]


def auroc(pos: list[float], neg: list[float]) -> float:
    """Chance that a random answerable question outscores a random unanswerable one (ties count half)."""
    if not pos or not neg:
        raise ValueError("need scores for both groups")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def sweep(answerable: list[tuple[float, bool]], unanswerable: list[float], thresholds) -> list[tuple]:
    """Per threshold t: share of answerable questions answered correctly / answered wrongly, and share
    of unanswerable questions that still get an answer (score >= t means 'answer', below means 'not found')."""
    n, m = len(answerable), len(unanswerable)
    return [(t, sum(s >= t and ok for s, ok in answerable) / n, sum(s >= t and not ok for s, ok in answerable) / n,
             sum(s >= t for s in unanswerable) / m) for t in thresholds]


def collect(index, chunks: list[str], qa, unanswerable):
    """Top-1 score and correctness for answerable questions, top-1 score for unanswerable ones."""
    top = lambda q: index.search(q, k=1)[0]  # noqa: E731
    answerable = [(s, fact in chunks[i]) for (q, fact) in qa for i, s in [top(q)]]
    return answerable, [top(q)[1] for q in unanswerable]


if __name__ == "__main__":
    chunks = by_sentence(CORPUS, 160)
    docs, query = bge_small()
    fused = HybridIndex([BM25(chunks), CharNgramIndex(chunks), DenseIndex(chunks, docs, query)], len(chunks))
    reranked = RerankIndex(fused, chunks, minilm(), pool=5)
    print(f"{len(chunks)} chunks, {len(QA) + len(PARAPHRASED)} answerable, {len(UNANSWERABLE)} unanswerable questions")
    for label, idx in (("rrf score", fused), ("cross-encoder p", reranked)):
        ans, una = collect(idx, chunks, QA + PARAPHRASED, UNANSWERABLE)
        lo = min(s for s, _ in ans)
        print(f"\n{label}: AUROC={auroc([s for s, _ in ans], una):.2f}  answerable min={lo:.3f}  unanswerable max={max(una):.3f}")
        ts = sorted({round(s, 3) for s in una})[-3:] if label == "rrf score" else (0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9)
        for t, ok, wrong, false in sweep(ans, una, ts):
            print(f"  t={t:<6} answered right={ok:.2f} answered wrong={wrong:.2f} unanswerable still answered={false:.2f}")
