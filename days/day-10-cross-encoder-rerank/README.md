# Day 10: Cross-encoder reranking on the fused top k

**Question:** Day 09's best first stage, RRF over BM25 + character n-grams + bge-small, still put the answer at rank 2 for one of 12 paraphrased questions. Does a cross-encoder that reads question and chunk together fix the order, how deep does the candidate pool need to be, and what does it cost per query?

This is step 2 of the RAG v2 track (roadmap row 1b). `rerank.py` adds `RerankIndex`: the first stage proposes `pool` chunks, **Xenova/ms-marco-MiniLM-L-6-v2** (ONNX on CPU via fastembed) scores each (question, chunk) pair, the pool is reordered and everything below it keeps the first-stage order. Same corpus, sentence packing (160 chars, 8 chunks) and 12 + 12 questions, through the unchanged Day 04 harness.

| Retriever | Original hit@1 / hit@3 / MRR@5 | Paraphrased hit@1 / hit@3 / MRR@5 | Rerank time per query (CPU) |
| --- | --- | --- | --- |
| RRF(BM25 + char + dense) | 1.00 / 1.00 / 1.00 | 0.92 / 1.00 / 0.96 | 0 |
| + rerank, pool = 3 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | about 28 to 42 ms |
| + rerank, pool = 5 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 50 to 300 ms (noisy run) |
| + rerank, pool = 8 (all chunks) | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | about 90 ms |

**A bug worth keeping:** the first run reported paraphrase hit@1 = 0.33 after reranking, which is impossible when only the top 3 are reordered. The cross-encoder returns raw logits, many of them negative for correct but paraphrased pairs, and the Day 04 harness (written for BM25 and cosine) counts a hit only when the score is above 0. Mapping logits through a sigmoid keeps the order and fixed it; a test now covers negative logits. Any evaluation harness that filters on `score > 0` silently breaks when the scorer changes.

**Takeaway:** the reranker moved the last paraphrased answer from rank 2 to rank 1, so every metric reached 1.00, and a pool of 3 was already enough because the first stage had every answer in its top 3. Cost grows with the pool, so the pool should be set by the first stage's Recall@k (Day 07), not by habit. With 8 chunks and 12 questions one question moves a metric by 0.08, and CPU timings vary between runs, so this shows the mechanism, not a benchmark.

**Next (RAG v2):** a larger question set from real documents, where the first stage misses more often and the pool depth really matters.
