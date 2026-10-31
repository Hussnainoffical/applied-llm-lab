import pytest
from hybrid import CORPUS, PARAPHRASED, BM25, CharNgramIndex, HybridIndex, by_sentence, char_ngrams, evaluate, rrf


def test_rrf_matches_hand_computed_scores():
    fused = rrf([[0, 1], [1, 2]], k=60)
    assert [d for d, _ in fused] == [1, 0, 2]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_char_ngrams_mark_word_boundaries():
    grams = char_ngrams("PDFs")
    assert " pd" in grams and "fs " in grams and " pdf" in grams


def test_char_ngrams_match_a_plural_that_bm25_misses():
    docs = ["PDF files are converted", "tables lose structure"]
    assert BM25(docs).search("PDFs", 1)[0][1] == 0
    i, score = CharNgramIndex(docs).search("PDFs", 1)[0]
    assert i == 0 and score > 0


def test_hybrid_drops_chunks_no_retriever_matched():
    docs = ["alpha beta", "gamma delta"]
    assert HybridIndex([BM25(docs), CharNgramIndex(docs)], 2).search("xyz", 3) == []


def test_fusion_lifts_paraphrase_hit_at_3_over_bm25_on_sentence_chunks():
    chunks = by_sentence(CORPUS, 160)
    bm = BM25(chunks)
    hybrid = HybridIndex([bm, CharNgramIndex(chunks)], len(chunks))
    a, b = evaluate(bm, chunks, PARAPHRASED), evaluate(hybrid, chunks, PARAPHRASED)
    assert (a["hit@3"], b["hit@3"]) == (pytest.approx(10 / 12), 1.0)
    assert b["hit@1"] == a["hit@1"] and b["mrr@5"] > a["mrr@5"]
