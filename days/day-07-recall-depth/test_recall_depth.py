from recall_depth import CORPUS, PARAPHRASED, QA, BM25, CharNgramIndex, HybridIndex, by_sentence, ceiling, fixed_size, recall_curve, saturation_depth


class FixedRanking:
    def __init__(self, order):
        self.order = order

    def search(self, query, k=5):
        return [(i, 1.0) for i in self.order[:k]]


def test_recall_curve_counts_rank_of_whole_fact_chunk():
    chunks = ["nothing here", "still nothing", "the fact is here"]
    assert recall_curve(FixedRanking([0, 1, 2]), chunks, [("q", "the fact is here")], 3) == [0.0, 0.0, 1.0]


def test_saturation_depth():
    assert saturation_depth([0.5, 0.75, 0.75, 1.0, 1.0]) == 4


def test_fixed_chunks_cap_recall_at_the_ceiling():
    chunks = fixed_size(CORPUS, 120)
    bm = BM25(chunks)
    assert ceiling(chunks, QA) == 8 / 12
    assert recall_curve(bm, chunks, QA, len(chunks))[-1] <= ceiling(chunks, QA)


def test_sentence_chunks_reach_full_recall_on_paraphrases_with_rrf():
    chunks = by_sentence(CORPUS, 160)
    bm, ch = BM25(chunks), CharNgramIndex(chunks)
    curve = recall_curve(HybridIndex([bm, ch], len(chunks)), chunks, PARAPHRASED, len(chunks))
    assert ceiling(chunks, QA) == 1.0
    assert curve[-1] == 1.0 and saturation_depth(curve) <= 3
