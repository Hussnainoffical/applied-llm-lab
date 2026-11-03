# Day 08: Validating LLM JSON output

**Question:** when a model is asked for JSON, how much does a strict `json.loads` reject that is only formatting noise, and how much does it accept that is still wrong?

`json_guard.py` compares three gates on 12 hand-written raw outputs, one per common failure mode when small models are asked for JSON (`cases.py`): code fences, prose around the object, trailing commas, truncation at the token limit, single quotes, missing keys, wrong types, an unknown stage name, empty question lists, and a refusal. The schema is an interview plan with named stages, like the one my InterviewMate model produces.

- **strict:** `json.loads` on the raw text
- **tolerant:** strip code fences and surrounding prose, drop trailing commas outside strings, then parse; it never adds braces, so a truncated object stays rejected
- **schema:** tolerant parse plus a check of keys, types, allowed stage names and non-empty questions

| Gate | Parsed / accepted | Notes |
| --- | --- | --- |
| strict | 6 / 12 parsed | rejects 3 harmless formatting cases (fence, prose, trailing comma), yet 4 of its 6 parsed objects are schema-invalid |
| tolerant | 9 / 12 parsed | recovers the 3 formatting cases; truncated, single-quoted and refusal outputs stay rejected |
| tolerant + schema | 5 / 12 accepted | matches the expected verdict on 12 / 12 cases |

**Takeaway:** parsing and validating are two separate gates. Strict parsing fails in both directions: it throws away usable output over formatting and lets through objects with a missing key, a string instead of a list, an invented stage or empty questions. A first regex version of the trailing-comma fix also changed text inside strings (`"x,]"` became `"x]"`); a small scanner that tracks string state fixed it, and a test now guards it. The 12 cases are hand-written to cover failure modes, not a sample of real model outputs, so the counts show the mechanism, not real-world rates.

**Next:** retry with the validation errors fed back to the model, and count how many retries each failure mode needs.
