"""Day 08: validating LLM JSON output - strict parse vs tolerant extraction + schema check.

Models asked for JSON often wrap it in prose or code fences, add trailing commas, or stop
mid-object when they hit max_new_tokens. A strict json.loads rejects all of these. A tolerant
extractor recovers the formatting cases, but it must never "repair" a truncated object, and a
parsed object can still be wrong (missing keys, wrong types, unknown values), so a schema check
has to follow. Schema here: an interview plan with stages, as in my InterviewMate project.

Run:  python json_guard.py
Test: python -m pytest -q
"""
import json
import re

STAGES = {"warm_opening", "cv_walkthrough", "project_deep_dive", "technical_skill_probing",
          "problem_solving_debugging", "behavioral_communication", "closing"}
SCHEMA = {"candidate": str, "stages": list}
STAGE_SCHEMA = {"stage": str, "questions": list}


def drop_trailing_commas(s: str) -> str:
    """Remove commas directly before } or ], but only outside strings (a regex would also edit "x,]")."""
    out, in_str, esc = [], False, False
    for i, ch in enumerate(s):
        if in_str:
            in_str, esc = (ch != '"' or esc), (ch == "\\" and not esc)
        elif ch == '"':
            in_str = True
        elif ch == "," and s[i + 1:].lstrip()[:1] in ("}", "]"):
            continue
        out.append(ch)
    return "".join(out)


def extract_json(text: str):
    """Return the parsed object or None. Strips code fences and surrounding prose, drops trailing commas.
    Never adds braces, so a truncated object stays unparseable."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        return json.loads(drop_trailing_commas(text[start:end + 1]))
    except json.JSONDecodeError:
        return None


def validate(obj) -> list[str]:
    """Schema errors for an interview plan; an empty list means valid."""
    if not isinstance(obj, dict):
        return ["not an object"]
    errors = [f"missing or wrong type: {k}" for k, t in SCHEMA.items() if not isinstance(obj.get(k), t)]
    for i, s in enumerate(obj.get("stages") or []):
        if not isinstance(s, dict) or any(not isinstance(s.get(k), t) for k, t in STAGE_SCHEMA.items()):
            errors.append(f"stage {i}: bad shape")
        elif s["stage"] not in STAGES:
            errors.append(f"stage {i}: unknown stage {s['stage']!r}")
        elif not s["questions"] or not all(isinstance(q, str) and q.strip() for q in s["questions"]):
            errors.append(f"stage {i}: empty or non-text questions")
    return errors


def strict(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    from cases import CASES
    rows = []
    for name, raw, should_pass in CASES:
        s, t = strict(raw), extract_json(raw)
        ok = t is not None and not validate(t)
        rows.append((name, s is not None, t is not None, ok, should_pass))
        print(f"{name:24} strict={s is not None!s:5} tolerant={t is not None!s:5} valid={ok!s:5} expected={should_pass}")
    n = len(rows)
    print(f"\nstrict parse: {sum(r[1] for r in rows)}/{n}  tolerant parse: {sum(r[2] for r in rows)}/{n}  "
          f"accepted after schema: {sum(r[3] for r in rows)}/{n}  matches expected: {sum(r[3] == r[4] for r in rows)}/{n}")
