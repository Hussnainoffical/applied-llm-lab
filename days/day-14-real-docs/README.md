# Day 14: a real question set from real documents

RAG v2, step 6. Every number so far came from a toy corpus I wrote myself, where the question words
and the answer words overlap a lot. Here the corpus is six sections of the Python Language Reference
that ship with CPython (`pydoc_data.topics`: try, raise, for, while, import, with; about 25k
characters with grammar rules, code examples and "Changed in version" notes). 16 questions are written
the way a user asks them ("Why is my exception variable gone after the handler finishes?"), each with
one exact answer span from the docs. Same retrievers as Days 02-05, top 10, 400-character chunks.

## Result (measured 12 Nov 2026, CPU)

| Chunking | Retriever | hit@1 | Recall@5 | MRR@10 | Not in top 10 |
|---|---|---|---|---|---|
| fixed 400 (68 chunks, ceiling 0.88) | BM25 | 0.19 | 0.56 | 0.34 | 5 |
| | char n-gram | 0.25 | 0.69 | 0.42 | 3 |
| | RRF (BM25 + char) | 0.38 | 0.62 | 0.48 | 3 |
| sentence packing 400 (77 chunks, ceiling 1.00) | BM25 | 0.12 | 0.69 | 0.36 | 2 |
| | char n-gram | 0.19 | 0.75 | 0.45 | 2 |
| | RRF (BM25 + char) | 0.25 | 0.81 | 0.46 | 2 |

## What I take from it

- The toy corpus flattered everything. On Day 05 RRF on sentence chunks had paraphrase hit@3 1.00; on
  real docs the same setup puts the answer first for only 4 of 16 questions, and in the top 5 for 13.
- Six sections on related statements share most of their vocabulary ("except", "exception", "clause",
  "suite"), so many chunks score close together and rank 1 becomes a coin toss. Recall@5 holds up much
  better than hit@1, which is the case for a reranker on a deeper pool, not for a different first stage.
- Fixed-size chunks cut 2 of 16 answer spans in half (ceiling 0.88), sentence packing kept all 16 whole.
  Fixed chunks still had the best hit@1 with RRF (0.38), so chunking and ranking have to be judged
  separately: ceiling first, then rank.
- Both sparse retrievers miss the same two questions on sentence chunks: "Does a bare except have to be
  the final handler?" (the docs say "expression-less except clause ... must be last") and "Where does the
  original error end up when I chain with from?" ("attached ... as the __cause__ attribute"). No word
  overlap, so this is where learned embeddings (Day 09) have to earn their place.
- Caveat: 16 questions written by one person against one document type. Next: dense retrieval and the
  Day 10 cross-encoder on this set, and a top-20 pool for the reranker.

Run: `python real_docs.py`, test: `python -m pytest -q` (4 tests, no network).
