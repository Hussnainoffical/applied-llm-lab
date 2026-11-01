import math

import pytest
from ndcg import CORPUS, QA, BM25, by_sentence, dcg, fixed_size, grade, mean_ndcg, ndcg_at_k


def test_grade_whole_half_and_none():
    fact = "Tables often lose their row structure in this step"
    assert grade("x. Tables often lose their row structure in this step. y", fact) == 2
    assert grade("Tables often lose their row str", fact) == 1
    assert grade("Embeddings map each chunk to a dense vector", fact) == 0


def test_dcg_matches_hand_computed_value():
    assert dcg([2, 0, 1]) == pytest.approx(3 / math.log2(2) + 0 + 1 / math.log2(4))


class FixedRanking:
    def __init__(self, order):
        self.order = order

    def search(self, query, k=5):
        return [(i, 1.0) for i in self.order[:k]]


def test_ndcg_is_one_for_ideal_order_and_lower_when_swapped():
    chunks = ["the fact is here", "the fact", "nothing"]
    assert ndcg_at_k(FixedRanking([0, 1, 2]), chunks, "q", "the fact is here") == pytest.approx(1.0)
    assert ndcg_at_k(FixedRanking([2, 1, 0]), chunks, "q", "the fact is here") < 1.0


def test_standard_ndcg_hides_facts_that_fixed_chunks_split():
    chunks = fixed_size(CORPUS, 120)
    split = sum(1 for _, f in QA if max(grade(c, f) for c in chunks) < 2)
    bm = BM25(chunks)
    assert split == 4
    assert mean_ndcg(bm, chunks, QA) == pytest.approx(1.0)
    assert mean_ndcg(bm, chunks, QA, fixed_ideal=True) < 0.8


def test_both_ndcg_versions_agree_on_sentence_chunks():
    chunks = by_sentence(CORPUS, 160)
    bm = BM25(chunks)
    assert mean_ndcg(bm, chunks, QA) == pytest.approx(mean_ndcg(bm, chunks, QA, fixed_ideal=True))
