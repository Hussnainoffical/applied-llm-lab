import numpy as np
import pytest
from cosine import PARAPHRASES, QA, SAMPLE, CosineIndex, by_sentence, top1_hits

DOCS = ["the cat sat on the mat", "dogs chase cats in the park", "stock prices fell sharply today"]


def test_relevant_doc_ranks_first():
    assert CosineIndex(DOCS).search("cat mat", 1)[0][0] == 0


def test_query_equal_to_doc_scores_one():
    assert CosineIndex(DOCS).search(DOCS[2], 1)[0][1] == pytest.approx(1.0)


def test_no_overlap_scores_zero():
    assert CosineIndex(DOCS).search("quantum physics", 1)[0][1] == 0


def test_rows_are_unit_length():
    assert np.allclose(np.linalg.norm(CosineIndex(DOCS).matrix, axis=1), 1.0)


def test_empty_corpus_rejected():
    with pytest.raises(ValueError):
        CosineIndex([])


def test_sentence_chunks_answer_all_original_questions():
    chunks = by_sentence(SAMPLE, 160)
    assert top1_hits(CosineIndex(chunks), chunks, QA) == len(QA)


def test_paraphrases_defeat_lexical_vectors():
    chunks = by_sentence(SAMPLE, 160)
    assert top1_hits(CosineIndex(chunks), chunks, PARAPHRASES) == 0
