# Day 16: section-aware chunks with a title prefix

RAG v2, step 8. On Day 14 sentence packing ran over the whole corpus, so some chunks start in one
statement's section and end in the next, and most chunks never say which statement they describe
("the else clause" can belong to try, for or while). Here each of the six Python Language Reference
sections is packed on its own (400 characters) and every chunk starts with the section title
(`The "try" statement. `). Same 16 questions, same lexical retrievers, top 10, no model.

## Result (measured 14 Nov 2026, CPython 3.13 on Windows)

| Chunking | Retriever | hit@1 | Recall@5 | MRR@10 | Not in top 10 |
|---|---|---|---|---|---|
| whole corpus (Day 14), 77 chunks | BM25 | 0.12 | 0.69 | 0.36 | 2 |
| | char n-grams | 0.19 | 0.75 | 0.45 | 2 |
| | RRF | 0.25 | 0.81 | 0.46 | 2 |
| per section, 78 chunks | BM25 | 0.12 | 0.62 | 0.38 | 2 |
| | char n-grams | 0.19 | 0.75 | 0.45 | 2 |
| | RRF | 0.38 | 0.81 | 0.54 | 2 |
| per section + title, 78 chunks | BM25 | 0.25 | 0.62 | 0.43 | 3 |
| | char n-grams | 0.38 | 0.75 | 0.55 | 2 |
| | RRF | 0.44 | 0.88 | 0.58 | 2 |

## What I take from it

- A one-line title on every chunk moved lexical RRF from hit@1 0.25 to 0.44 and Recall@5 0.81 to 0.88,
  the same hit@1 the Day 15 cross-encoder reached, with no model and no extra latency.
- Most of the gain is the title, not the boundaries: per-section chunking alone helped RRF (0.25 -> 0.38)
  but left BM25 and char n-grams where they were.
- It is not free for every retriever: the repeated title adds the same words to every chunk of a
  section, and BM25 lost one answer from its top 10 (Recall@5 0.69 -> 0.62). The fused ranking hides that.
- The two questions with no word overlap are still missed by every lexical setup; that is a job for
  dense retrieval or query rewriting, not for chunk metadata.
- Results depend on the CPython version that ships pydoc_data.topics (a Linux 3.x build gave 69 chunks
  and a lower ceiling), so the table is pinned to the version above.

Run: `python section_context.py` · Test: `python -m pytest -q`
