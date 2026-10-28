import pytest
from bm25 import BM25, tokenize

DOCS = ["the cat sat on the mat", "dogs chase cats in the park", "stock prices fell sharply today"]


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Cat, and THE mat!") == ["cat", "mat"]


def test_relevant_doc_ranks_first():
    assert BM25(DOCS).search("cat mat", 1)[0][0] == 0


def test_no_overlap_scores_zero():
    assert BM25(DOCS).score("quantum", 2) == 0


def test_rare_term_weighs_more_than_common_term():
    bm = BM25(["apple banana", "apple cherry", "apple durian"])
    assert bm.idf["banana"] > bm.idf["apple"]


def test_longer_doc_penalised_for_same_tf():
    bm = BM25(["rag", "rag filler filler filler filler"])
    assert bm.score("rag", 0) > bm.score("rag", 1)


def test_empty_corpus_rejected():
    with pytest.raises(ValueError):
        BM25([])
