import numpy as np

from dense_rerank import CORPUS, QA, by_sentence, pipeline, run


def bag(texts):  # stands in for the embedder: word-count vectors over a tiny vocabulary
    vocab = ["exception", "loop", "import", "with", "else", "raise"]
    return np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype=float)


def overlap(query, docs):  # stands in for the cross-encoder: shared words, can be zero
    q = set(query.lower().split())
    return [len(q & set(d.lower().split())) - 1.0 for d in docs]


def test_pipeline_has_every_stage_on_the_same_chunks():
    chunks = by_sentence(CORPUS, 400)
    names = list(pipeline(chunks, bag, bag, overlap, pools=(5, 20)))
    assert names == ["rrf(bm25+char)", "dense", "rrf(bm25+char+dense)", "+ rerank pool=5", "+ rerank pool=20"]


def test_run_reports_fractions_and_latency():
    chunks = by_sentence(CORPUS, 400)
    for s in run(chunks, pipeline(chunks, bag, bag, overlap, pools=(20,))).values():
        assert 0 <= s["hit@1"] <= s["recall@5"] <= 1 and s["ms/query"] >= 0


def test_oracle_reranker_lifts_every_pooled_answer_to_rank_one():
    chunks = by_sentence(CORPUS, 400)
    flat = [" ".join(c.split()) for c in chunks]
    spans = [f for _, f in QA]

    def oracle(query, docs):  # 1 for a chunk holding this question's answer span, else -1
        f = spans[[q for q, _ in QA].index(query)]
        return [1.0 if f in " ".join(d.split()) else -1.0 for d in docs]

    rows = run(chunks, pipeline(chunks, bag, bag, oracle, pools=(len(chunks),)))
    reachable = sum(any(f in c for c in flat) for f in spans) / len(QA)
    assert rows[f"+ rerank pool={len(chunks)}"]["hit@1"] == reachable
