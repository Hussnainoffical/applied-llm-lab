"""Day 15: learned embeddings and a cross-encoder on the real question set (RAG v2, step 3).

Day 14 moved to real documents (six Python Language Reference sections, 16 user-style questions)
and lexical RRF reached hit@1 0.25, Recall@5 0.81 on sentence chunks. Here the Day 09 dense
retriever (bge-small-en-v1.5) joins the fusion, and the Day 10 cross-encoder (ms-marco-MiniLM-L-6-v2)
reorders the fused top `pool` chunks, so every stage is measured on the same chunks and questions.

Run:  python dense_rerank.py   (ONNX models via fastembed, CPU)
Test: python -m pytest -q
"""
import sys
import time
from pathlib import Path

for day in ("day-04-hit-rate-mrr", "day-05-hybrid-rrf", "day-09-dense-vs-sparse", "day-10-cross-encoder-rerank", "day-14-real-docs"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / day))
from dense import DenseIndex, bge_small  # noqa: E402
from hybrid import BM25, CharNgramIndex, HybridIndex  # noqa: E402
from real_docs import CORPUS, QA, by_sentence, ceiling, scores  # noqa: E402
from rerank import RerankIndex, minilm  # noqa: E402


def pipeline(chunks: list[str], embed_docs, embed_query, score, pools=(5, 20)) -> dict:
    """Name -> retriever, from lexical fusion up to fusion + dense + reranking."""
    n = len(chunks)
    bm, ch, de = BM25(chunks), CharNgramIndex(chunks), DenseIndex(chunks, embed_docs, embed_query)
    lexical, full = HybridIndex([bm, ch], n), HybridIndex([bm, ch, de], n)
    out = {"rrf(bm25+char)": lexical, "dense": de, "rrf(bm25+char+dense)": full}
    out.update({f"+ rerank pool={p}": RerankIndex(full, chunks, score, p) for p in pools})
    return out


def run(chunks: list[str], retrievers: dict) -> dict:
    rows = {}
    for name, idx in retrievers.items():
        t = time.perf_counter()
        s = scores(idx, chunks)
        rows[name] = {**s, "ms/query": 1000 * (time.perf_counter() - t) / len(QA)}
    return rows


if __name__ == "__main__":
    chunks = by_sentence(CORPUS, 400)
    docs, query = bge_small()
    print(f"sentence packing 400: {len(chunks)} chunks, ceiling {ceiling(chunks):.2f}")
    for name, s in run(chunks, pipeline(chunks, docs, query, minilm())).items():
        print(f"  {name:<22} hit@1={s['hit@1']:.2f} R@5={s['recall@5']:.2f} MRR@10={s['mrr@10']:.2f} "
              f"missed={len(s['missed'])} {s['ms/query']:.0f} ms/query")
