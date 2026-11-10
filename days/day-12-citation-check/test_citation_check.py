from citation_check import CHUNKS, FABRICATED, SUPPORTED, exact, overlap, overlap_ids, recall, same_sentence, score


def test_corpus_has_eight_sentence_chunks():
    assert len(CHUNKS) == 8


def test_recall_ignores_stopwords():
    assert recall("The phases.", "two phases") == 1.0


def test_exact_needs_the_literal_sentence():
    assert exact("BM25 rewards exact term matches.", CHUNKS[4])
    assert not exact("Each chunk keeps its file name and page number as metadata.", CHUNKS[2])


def test_word_recall_lets_a_changed_number_through():
    claim, i = FABRICATED[1]  # 768 instead of 384
    assert overlap(claim, CHUNKS[i]) and not overlap_ids(claim, CHUNKS[i])


def test_one_sentence_rule_catches_entity_swap_but_not_role_swap():
    assert not same_sentence("Dense search rewards exact term matches.", CHUNKS[4])
    assert same_sentence("Dense search handles rare identifiers but can miss synonyms.", CHUNKS[4])


def test_every_check_catches_every_wrong_chunk_citation():
    for check in (exact, overlap, overlap_ids, same_sentence):
        assert score(check)["wrong chunk caught"] == 1.0


def test_one_sentence_rule_scores():
    s = score(same_sentence)
    assert s["supported accepted"] == 11 / 12 and s["fabricated caught"] == 8 / 9
    assert len(SUPPORTED) == 12
