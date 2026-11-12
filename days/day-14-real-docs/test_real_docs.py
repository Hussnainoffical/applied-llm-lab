from real_docs import CORPUS, QA, by_sentence, ceiling, scores


def test_every_answer_span_is_in_the_real_docs():
    flat = " ".join(CORPUS.split())
    assert [f for _, f in QA if f not in flat] == []


def test_questions_do_not_copy_the_answer_span():
    assert all(f.lower() not in q.lower() for q, f in QA)


def test_ceiling_counts_whole_spans_only():
    assert ceiling(["the saved exception", "is discarded"], [("q", "the saved exception is discarded")]) == 0.0
    assert ceiling(by_sentence(CORPUS, 10_000)) == 1.0


class FixedRanking:
    def search(self, query, k=10):
        return [(0, 1.0), (1, 1.0)][:k]


def test_scores_use_rank_of_first_chunk_with_span():
    s = scores(FixedRanking(), ["nothing", "the fact"], [("q", "the fact")])
    assert (s["hit@1"], s["recall@5"], s["mrr@10"]) == (0.0, 1.0, 0.5)
