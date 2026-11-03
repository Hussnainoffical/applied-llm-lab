"""Hand-written raw outputs, one per common failure mode when small LLMs are asked for JSON.
(name, raw text, should be accepted after extraction + schema check)"""
GOOD = '{"candidate": "A. Example", "stages": [{"stage": "warm_opening", "questions": ["Tell me about yourself."]}, {"stage": "closing", "questions": ["Any questions for us?"]}]}'

CASES = [
    ("clean", GOOD, True),
    ("code_fence", "```json\n" + GOOD + "\n```", True),
    ("prose_before_after", "Here is the interview plan:\n" + GOOD + "\nGood luck!", True),
    ("trailing_comma", GOOD.replace('"Any questions for us?"]', '"Any questions for us?",]'), True),
    ("pretty_printed", GOOD.replace(", ", ",\n  "), True),
    ("truncated", GOOD[:120], False),
    ("single_quotes", GOOD.replace('"', "'"), False),
    ("missing_candidate", GOOD.replace('"candidate": "A. Example", ', ""), False),
    ("stages_as_string", '{"candidate": "A. Example", "stages": "warm_opening, closing"}', False),
    ("unknown_stage", GOOD.replace("warm_opening", "salary_negotiation"), False),
    ("empty_questions", GOOD.replace('["Tell me about yourself."]', "[]"), False),
    ("no_json_at_all", "Sorry, I cannot create an interview plan for this CV.", False),
]
