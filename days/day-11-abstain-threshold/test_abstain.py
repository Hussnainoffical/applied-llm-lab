import pytest

from abstain import QA, PARAPHRASED, UNANSWERABLE, auroc, collect, sweep


class Fixed:
    """Fake index: each query maps to a fixed (chunk id, score) top hit."""

    def __init__(self, table):
        self.table = table

    def search(self, query, k=1):
        return [self.table[query]][:k]


def test_auroc_perfect_reversed_and_ties():
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.1], [0.9]) == 0.0
    assert auroc([0.5, 0.5], [0.5]) == 0.5


def test_auroc_needs_both_groups():
    with pytest.raises(ValueError):
        auroc([], [0.1])


def test_sweep_counts_right_wrong_and_false_answers():
    answerable = [(0.9, True), (0.6, False), (0.2, True), (0.05, True)]
    rows = sweep(answerable, [0.7, 0.1], [0.5, 0.1])
    assert rows[0] == (0.5, 0.25, 0.25, 0.5)
    assert rows[1] == (0.1, 0.5, 0.25, 1.0)


def test_collect_checks_the_fact_in_the_top_chunk():
    chunks = ["the sky is blue", "grass is green"]
    idx = Fixed({"sky?": (0, 0.8), "grass?": (0, 0.4), "moon?": (1, 0.3)})
    ans, una = collect(idx, chunks, [("sky?", "sky is blue"), ("grass?", "grass is green")], ["moon?"])
    assert ans == [(0.8, True), (0.4, False)] and una == [0.3]


def test_unanswerable_set_is_separate_from_the_labelled_questions():
    asked = {q for q, _ in QA + PARAPHRASED}
    assert len(UNANSWERABLE) == 12 == len(set(UNANSWERABLE)) and not asked & set(UNANSWERABLE)
