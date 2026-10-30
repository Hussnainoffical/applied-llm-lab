import pytest
from evaluate import CORPUS, PARAPHRASED, QA, BM25, CosineIndex, answer_rank, by_sentence, evaluate, fixed_size


class FixedIndex:
    """Returns a fixed ranking, so the metrics can be checked by hand."""

    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, query, k=3):
        return [(i, 1.0) for i in self.ranking[:k]]


CHUNKS = ["alpha fact here", "beta fact here", "gamma fact here"]


def test_rank_is_one_based_position_of_answer_chunk():
    assert answer_rank(FixedIndex([2, 1, 0]), CHUNKS, "q", "beta fact") == 2


def test_missing_answer_gives_none():
    assert answer_rank(FixedIndex([0, 2]), CHUNKS, "q", "beta fact", k=2) is None


def test_mrr_rewards_rank_while_hit_at_3_does_not():
    qa = [("q1", "alpha fact"), ("q2", "beta fact")]
    m = evaluate(FixedIndex([2, 1, 0]), CHUNKS, qa)
    assert m["hit@3"] == 1.0 and m["hit@1"] == 0.0
    assert m["mrr@5"] == pytest.approx((1 / 3 + 1 / 2) / 2)


def test_every_fact_is_in_the_corpus():
    flat = " ".join(CORPUS.split())
    assert all(f in flat for _, f in QA + PARAPHRASED)


def test_sentence_chunks_answer_every_original_question_first():
    chunks = by_sentence(CORPUS, 160)
    assert evaluate(BM25(chunks), chunks)["hit@1"] == 1.0


def test_overlap_raises_hit_at_3_but_lowers_hit_at_1_on_paraphrases():
    plain, overlap = fixed_size(CORPUS, 120), fixed_size(CORPUS, 120, 30)
    a, b = evaluate(CosineIndex(plain), plain, PARAPHRASED), evaluate(CosineIndex(overlap), overlap, PARAPHRASED)
    assert b["hit@3"] > a["hit@3"] and b["hit@1"] < a["hit@1"]
