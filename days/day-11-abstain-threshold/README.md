# Day 11: when should retrieval say "not found"?

RAG v2, step 3. Days 09 and 10 measured how well the right chunk gets to rank 1. This day asks the
opposite question: what happens when the corpus has no answer at all? Twelve on-topic questions the
corpus cannot answer (GPU for training, FAISS licence, context window size, ...) are added to the 24
labelled questions (12 original + 12 paraphrased). For each question only the top-1 hit and its score
are kept, and two scores are tested as an abstention rule: "answer if score >= t, else say not found".

- **RRF score** of the fused BM25 + character n-gram + bge-small list (Day 09)
- **Cross-encoder probability** after MiniLM reranks the fused top 5 (Day 10, sigmoid of the logit)

## Result (8 chunks, CPU, measured 6 Nov 2026)

| Signal | AUROC answerable vs unanswerable | Best cut-off seen | Answered right | Unanswerable still answered |
|---|---|---|---|---|
| RRF score | 0.68 | t = 0.049 (top unanswerable score) | 0.79 | 0.42 |
| Cross-encoder p | 0.85 | t = 0.3 | 0.71 | 0.17 |
| Cross-encoder p | 0.85 | t = 0.9 | 0.62 | 0.00 |

No threshold in either sweep produced a wrong answer on an answerable question except RRF at its
lowest cut-off (0.04).

## What I take from it

- RRF only knows ranks. Its score still carries a little signal (0.68) because the three retrievers
  agree more often on real answers, but even the strictest useful cut-off lets 42% of unanswerable
  questions through. It cannot be the "not found" switch.
- The cross-encoder probability separates the two groups much better (0.85). A cut-off around 0.3 keeps
  most answers and blocks most unanswerable questions; at 0.9 nothing unanswerable gets through, but
  about a third of the answerable questions are refused too.
- The cut-off is a product decision, not a constant: refusing a real question and answering an
  unanswerable one have different costs, so the threshold belongs in the evaluation set, chosen per
  use case and re-checked when the corpus changes.
- Caveat: 8 chunks and 36 questions is a toy set; the numbers show the behaviour, not a benchmark.

Run: `python abstain.py`, test: `python -m pytest -q` (5 tests, no model needed).
