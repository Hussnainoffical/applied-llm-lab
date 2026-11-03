from cases import CASES, GOOD
from json_guard import extract_json, strict, validate


def test_every_case_matches_expected():
    for name, raw, should_pass in CASES:
        obj = extract_json(raw)
        assert (obj is not None and not validate(obj)) == should_pass, name


def test_truncated_output_is_never_repaired():
    for cut in range(1, len(GOOD) - 1):
        assert extract_json(GOOD[:cut]) is None


def test_strict_rejects_formatting_noise_that_tolerant_recovers():
    raw = "```json\n" + GOOD + "\n```"
    assert strict(raw) is None and extract_json(raw) == strict(GOOD)


def test_validate_reports_unknown_stage():
    obj = strict(GOOD.replace("closing", "small_talk"))
    assert validate(obj) == ["stage 1: unknown stage 'small_talk'"]


def test_trailing_comma_inside_string_is_kept():
    obj = extract_json('{"candidate": "x,]", "stages": [{"stage": "closing", "questions": ["ok"]}]}')
    assert obj["candidate"] == "x,]"


def test_escaped_quote_does_not_end_the_string():
    obj = extract_json('{"candidate": "say \\"hi,]\\"", "stages": [],}')
    assert obj == {"candidate": 'say "hi,]"', "stages": []}
