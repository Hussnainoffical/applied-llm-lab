# Day 15: dense retrieval and a cross-encoder on the real question set

RAG v2, step 7. Day 14 moved the harness to real documents (six Python Language Reference sections,
16 user-style questions) and lexical RRF on sentence chunks reached hit@1 0.25 and Recall@5 0.81.
Here the Day 09 dense retriever (BAAI/bge-small-en-v1.5, ONNX via fastembed) joins the fusion, and the
Day 10 cross-encoder (Xenova/ms-marco-MiniLM-L-6-v2) reorders the fused top 5 or top 20.
Same 77 sentence chunks (400 characters), same questions, top 10, CPU.

## Result (measured 13 Nov 2026, CPU)

| Retriever | hit@1 | Recall@5 | MRR@10 | Not in top 10 | ms/query |
|---|---|---|---|---|---|
| RRF (BM25 + char n-grams) | 0.25 | 0.81 | 0.46 | 2 | 11 |
| dense (bge-small) | 0.31 | 0.62 | 0.45 | 2 | 52 |
| RRF (BM25 + char + dense) | 0.31 | 0.88 | 0.55 | 2 | 46 |
| + rerank, pool 5 | 0.44 | 0.88 | 0.56 | 2 | 496 |
| + rerank, pool 20 | 0.44 | 0.75 | 0.53 | 3 | 2116 |

## What I take from it

- Dense alone gets more answers to rank 1 than the lexical pair (0.31 vs 0.25) but fewer into the top 5
  (0.62 vs 0.81). Fused, it lifts Recall@5 to 0.88, the best first stage so far on real text.
- The cross-encoder over the top 5 is the clear win: hit@1 0.25 -> 0.44 at the same recall.
- A bigger pool is not free and not better here: pool 20 costs about 4x the time and pushes answers
  the fusion had in its top 5 further down (Recall@5 0.88 -> 0.75, one answer out of the top 10).
  The general-domain MS MARCO model prefers chunks that look like an answer to chunks with the
  grammar-rule wording these docs use. A larger pool needs a stronger or domain-tuned reranker.

Run: `python dense_rerank.py` · Test: `python -m pytest -q` (tests use stand-in embedder and scorer, no download)