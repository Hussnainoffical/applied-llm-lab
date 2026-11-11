from nli_check import CHUNKS, FABRICATED, SUPPORTED, WRONG_CHUNK, cascade, score


def fake(premise: str, hypothesis: str) -> float:  # stands in for the model: "entailed" if the words are there
    return 1.0 if all(w in premise.lower() for w in hypothesis.lower().rstrip(".").split()) else 0.0


def test_cascade_skips_model_for_claims_the_rule_rejects():
    calls: list = []
    check = cascade(fake, calls=calls)
    assert not any(check(c, CHUNKS[i]) for c, i in WRONG_CHUNK)
    assert calls == []  # every wrong-chunk citation is stopped before the model


def test_cascade_needs_both_rule_and_model():
    assert not cascade(lambda p, h: 1.0)("Dense search rewards exact term matches.", CHUNKS[4])  # rule fails
    assert not cascade(lambda p, h: 0.1)(*(SUPPORTED[0][0], CHUNKS[SUPPORTED[0][1]]))  # model says no


def test_score_rates_are_fractions():
    s = score(cascade(lambda p, h: 0.9))
    assert set(s) == {"supported accepted", "wrong chunk caught", "fabricated caught"}
    assert s["wrong chunk caught"] == 1.0 and 0 <= s["fabricated caught"] < 1  # role swap passes rule + yes-model


def test_sets_unchanged_from_day_12():
    assert (len(SUPPORTED), len(WRONG_CHUNK), len(FABRICATED)) == (12, 12, 9)
