# Day 13: an NLI model only where the cheap checks pass

RAG v2, step 5. Day 12 found that the one-sentence rule ("every content word of the claim sits in one
sentence of the cited chunk") catches all wrong-chunk citations and 8 of 9 fabrications, but lets a
role swap with the same words through. The usual answer is an LLM judge or an entailment model on every
claim. Here a small NLI cross-encoder (`Xenova/nli-deberta-v3-xsmall`, ONNX on CPU, premise = cited
chunk, hypothesis = claim) runs on the same Day 12 sets, alone and behind the rule.

## Result (measured 11 Nov 2026, CPU)

| Check | Supported accepted | Wrong chunk caught | Fabricated caught | Model calls |
|---|---|---|---|---|
| One-sentence rule | 0.92 | 1.00 | 0.89 | 0 |
| NLI alone, p(entail) >= 0.5 | 1.00 | 1.00 | 0.78 | 33 |
| Rule, then NLI | 0.92 | 1.00 | 1.00 | 12 |

About 25 ms per model call on CPU.

## What I take from it

- The two checks fail on different claims. NLI alone accepted two entity swaps where both entities sit in
  the chunk: "Dense search rewards exact term matches" (p = 0.91) and "BM25 can miss rare identifiers"
  (p = 0.95). The chunk talks about BM25 and dense search in one passage, and the small model does not
  keep track of which method does what. The one-sentence rule rejects both.
- The role swap that passes the rule ("Dense search handles rare identifiers but can miss synonyms")
  gets p = 0.36 from the model, so the cascade catches all 9 fabrications.
- Order matters for cost: every wrong-chunk citation and 8 fabrications stop at the rule, so the model
  sees 12 of 33 claims. The supported claim the rule wrongly rejects ("the answer chunk" vs "that chunk")
  stays rejected; the model would have accepted it (p = 0.99), so a borderline rule failure could go to
  the model instead of straight to "unsupported".
- Changed numbers and names get p close to 0.00 from the model; one honest claim (PDF to plain text)
  only gets 0.71, so a threshold much above 0.7 would start rejecting supported sentences.
- Caveat: hand-written claims on a toy corpus, one small model; this shows how the failure modes
  combine, not production rates. Next: a real question set from real documents.

Run: `python nli_check.py` (downloads the ONNX model once), test: `python -m pytest -q` (4 tests, no model).
