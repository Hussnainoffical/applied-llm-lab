import pytest

from rerank import CORPUS, QA, BM25, RerankIndex, by_sentence


class Fixed:
    """Fake first stage that always returns the same ranking."""

    def __init__(self, order):
        self.order = order

    def search(self, query, k=3):
        return [(i, 1.0 / (r + 1)) for r, i in enumerate(self.order)][:k]


def by_length(query, docs):
    return [len(d) for d in docs]  # fake cross-encoder: longer chunk = more relevant


def test_rerank_reorders_only_the_pool_and_keeps_the_tail():
    chunks = ["aa", "a", "aaaa", "aaa"]
    hits = RerankIndex(Fixed([0, 1, 2, 3]), chunks, by_length, pool=2).search("q", k=4)
    assert [i for i, _ in hits] == [0, 1, 2, 3]
    hits = RerankIndex(Fixed([1, 0, 3, 2]), chunks, by_length, pool=3).search("q", k=4)
    assert [i for i, _ in hits] == [3, 0, 1, 2]


def test_pool_larger_than_k_still_returns_k_hits_and_counts_time():
    idx = RerankIndex(Fixed([0, 1, 2, 3]), ["a", "bb", "ccc", "dddd"], by_length, pool=4)
    hits = idx.search("q", k=2)
    assert [i for i, _ in hits] == [3, 2] and idx.seconds >= 0


def test_invalid_pool_is_rejected():
    with pytest.raises(ValueError):
        RerankIndex(Fixed([0]), ["a"], by_length, pool=0)


def test_works_on_the_real_corpus_with_bm25_first_stage():
    chunks = by_sentence(CORPUS, 160)
    hits = RerankIndex(BM25(chunks), chunks, by_length, pool=5).search(QA[0][0], k=3)
    assert len(hits) == 3 and len({i for i, _ in hits}) == 3


def test_negative_logits_still_count_as_hits_and_keep_their_order():
    hits = RerankIndex(Fixed([0, 1, 2]), ["a", "b", "c"], lambda q, d: [-4.0, -1.0, -9.0], pool=3).search("q", k=3)
    assert [i for i, _ in hits] == [1, 0, 2] and all(0 < s < 1 for _, s in hits)
