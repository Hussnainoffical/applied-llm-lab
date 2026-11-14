from section_context import QA, SECTIONS, ceiling, per_section, retrievers, scores


def test_no_chunk_mixes_two_sections():
    chunks = per_section(400, prefix=True)
    titles = [f'The "{s}" statement. ' for s in SECTIONS]
    assert all(sum(c.startswith(t) for t in titles) == 1 for c in chunks)


def test_prefix_keeps_every_answer_span_reachable():
    assert ceiling(per_section(400, prefix=True)) == ceiling(per_section(400))


def test_scores_are_valid_fractions():
    chunks = per_section(400, prefix=True)
    for idx in retrievers(chunks).values():
        s = scores(idx, chunks)
        assert 0 <= s["hit@1"] <= s["recall@5"] <= 1 and len(s["missed"]) <= len(QA)
