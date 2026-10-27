import pytest
from chunking import by_paragraph, by_sentence, fixed_size

TEXT = "One. Two two. Three three three.\n\nFour four four four."


def test_fixed_size_covers_text_without_overlap():
    assert "".join(fixed_size(TEXT, 10)) == TEXT


def test_overlap_repeats_characters():
    chunks = fixed_size("abcdefghij", 4, 2)
    assert chunks[:2] == ["abcd", "cdef"]


def test_bad_overlap_rejected():
    with pytest.raises(ValueError):
        fixed_size(TEXT, 5, 5)


def test_sentences_never_cut():
    for c in by_sentence(TEXT, 20):
        assert c.endswith(".")


def test_long_sentence_kept_whole():
    assert by_sentence("A very long sentence here.", 5) == ["A very long sentence here."]


def test_paragraphs():
    assert by_paragraph(TEXT) == ["One. Two two. Three three three.", "Four four four four."]
