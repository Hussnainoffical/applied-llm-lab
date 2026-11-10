# Day 12: does the cited chunk support the sentence?

RAG v2, step 4. Days 09 to 11 were about getting the right chunk to the model and saying "not found"
when there is none. This day looks at the other end: an answer sentence that ends in a citation. A
citation can point to the wrong chunk, or the chunk can say something slightly different (another
number, another method). Before paying for an LLM judge, how much do cheap deterministic checks catch?

Test set on the 8 sentence-packed chunks from Day 04:

- **12 supported claims**, reworded the way a model would write them, each citing the right chunk
- **12 wrong-chunk citations**: the same claims, cited to another chunk
- **9 fabricated claims** citing the topically right chunk: changed numbers (768 vs 384, three vs two),
  swapped entities (Pinecone, author name, "Dense search rewards exact term matches"), a negation, and
  one claim that uses exactly the chunk's words with the roles swapped

## Result (measured 10 Nov 2026, no model)

| Check | Supported accepted | Wrong chunk caught | Fabricated caught |
|---|---|---|---|
| Exact substring | 0.58 | 1.00 | 1.00 |
| Content-word recall >= 0.75 | 1.00 | 1.00 | 0.00 |
| Recall + numbers/identifiers must appear | 1.00 | 1.00 | 0.22 |
| All content words inside one sentence | 0.92 | 1.00 | 0.89 |

## What I take from it

- Wrong-chunk citations are the easy case: every check caught all 12, because a cited chunk about
  another topic shares too few words.
- Word overlap is a bad support signal. A claim with one changed number or entity still keeps 75 to 100% of
  its content words in the chunk, so recall >= 0.75 passed all 9 fabrications. Requiring digits and names to
  appear only helps when the change is a digit or a capitalised name ("three", "hundred", "never" slip by).
- Scoping the check to one sentence did most of the work: every content word must sit in the same
  sentence, so "Dense search rewards exact term matches" fails because "Dense" and "exact" live in
  different sentences. It refused one honest paraphrase ("the answer chunk" vs "that chunk").
- What no bag-of-words check can see is a role swap with the same words ("handles rare identifiers but
  can miss synonyms"). That one needs an entailment model or an LLM judge, which can now run only on
  the claims that pass the cheap checks.
- Caveat: hand-written claims on a toy corpus; this shows failure modes, not production rates.

Run: `python citation_check.py`, test: `python -m pytest -q` (7 tests, no model needed).
