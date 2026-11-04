import importlib.util

import numpy as np
import pytest

from dense import BM25, CORPUS, PARAPHRASED, QA, DenseIndex, HybridIndex, by_sentence, compare, unit

TOPICS = ["pdf", "vector", "bm25", "fusion", "rerank", "phase"]


def topic_embed(texts):
    """Fake embedder: one dimension per topic word, so ranking is predictable without a model."""
    return np.array([[t.lower().count(w) for w in TOPICS] for t in texts], dtype=float)


def test_unit_keeps_zero_rows_at_zero():
    assert np.allclose(unit(np.array([[3.0, 4.0], [0.0, 0.0]])), [[0.6, 0.8], [0.0, 0.0]])


def test_dense_index_ranks_by_cosine_and_uses_query_embedder():
    docs = ["about pdf files", "about vector search", "bm25 and fusion"]
    idx = DenseIndex(docs, topic_embed, lambda xs: topic_embed(["vector"]))
    top = idx.search("anything", k=3)
    assert top[0][0] == 1 and top[0][1] == pytest.approx(1.0) and top[1][1] == 0.0


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        DenseIndex([], topic_embed)


def test_rrf_with_dense_lists_each_chunk_once_with_positive_score():
    chunks = by_sentence(CORPUS, 160)
    hits = HybridIndex([BM25(chunks), DenseIndex(chunks, topic_embed)], len(chunks)).search(QA[0][0], k=len(chunks))
    ids = [i for i, _ in hits]
    assert ids and len(ids) == len(set(ids)) and all(s > 0 for _, s in hits)


def test_compare_reports_every_retriever_on_both_question_sets():
    res = compare(by_sentence(CORPUS, 160), topic_embed, topic_embed)
    assert set(res) == {"bm25", "char-ngram", "dense", "rrf(bm25+dense)", "rrf(bm25+char+dense)"}
    assert all(set(r) == {"original", "paraphrased"} for r in res.values())
    assert len(PARAPHRASED) == len(QA) == 12
