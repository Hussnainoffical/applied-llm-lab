"""Day 09: learned embeddings vs sparse retrievers on paraphrased questions (RAG v2, step 1).

Days 03 to 07 showed the limit of lexical matching: BM25 and TF-IDF scored 0 on paraphrases that
share no words with the answer, and character n-grams only rescued part of them. Here a small
learned embedding model (BAAI/bge-small-en-v1.5, 384 dims, ONNX via fastembed) joins the same
harness, alone and fused with BM25 through RRF, on the same corpus, chunkings and 12 + 12 questions.

Run:  python dense.py          (downloads the ONNX model on first use, CPU only)
Test: python -m pytest -q
"""
import sys
from pathlib import Path
from typing import Callable

import numpy as np

for day in ("day-04-hit-rate-mrr", "day-05-hybrid-rrf"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / day))
from evaluate import evaluate  # noqa: E402
from hybrid import CORPUS, PARAPHRASED, QA, BM25, CharNgramIndex, HybridIndex, by_sentence, fixed_size  # noqa: E402

Embed = Callable[[list[str]], np.ndarray]


def unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x, dtype=float), where=n > 0)


class DenseIndex:
    """Chunks embedded once and L2-normalised; a search is one matrix-vector product (cosine)."""

    def __init__(self, docs: list[str], embed_docs: Embed, embed_query: Embed | None = None):
        if not docs:
            raise ValueError("need at least one document")
        self.embed_query = embed_query or embed_docs
        self.matrix = unit(np.asarray(embed_docs(docs), dtype=float))

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        scores = self.matrix @ unit(np.asarray(self.embed_query([query]), dtype=float))[0]
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in top]


def bge_small() -> tuple[Embed, Embed]:
    """Document and query embedders; query_embed adds the model's retrieval instruction."""
    from fastembed import TextEmbedding
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return (lambda xs: np.array(list(model.embed(xs))), lambda xs: np.array(list(model.query_embed(xs))))


def compare(chunks: list[str], embed_docs: Embed, embed_query: Embed) -> dict:
    bm, ch, de = BM25(chunks), CharNgramIndex(chunks), DenseIndex(chunks, embed_docs, embed_query)
    n = len(chunks)
    retrievers = {"bm25": bm, "char-ngram": ch, "dense": de,
                  "rrf(bm25+dense)": HybridIndex([bm, de], n), "rrf(bm25+char+dense)": HybridIndex([bm, ch, de], n)}
    return {name: {qs: evaluate(idx, chunks, q) for qs, q in (("original", QA), ("paraphrased", PARAPHRASED))}
            for name, idx in retrievers.items()}


if __name__ == "__main__":
    docs, query = bge_small()
    for name, chunks in [("fixed_size 120", fixed_size(CORPUS, 120)), ("sentence packing 160", by_sentence(CORPUS, 160))]:
        print(f"{name}: {len(chunks)} chunks")
        for retriever, res in compare(chunks, docs, query).items():
            for qs, m in res.items():
                print(f"  {retriever:<21} {qs:<12} hit@1={m['hit@1']:.2f} hit@3={m['hit@3']:.2f} mrr@5={m['mrr@5']:.2f}")
