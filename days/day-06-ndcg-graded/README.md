# Day 06: nDCG@5 with graded relevance

**Question:** hit rate is binary. Does a graded metric like nDCG tell a different story, and can it compare chunking strategies?

`ndcg.py` labels every chunk per question with a grade: **2** if it holds the whole fact, **1** if it holds a contiguous piece of at least half of the fact (`difflib` longest match), **0** otherwise. nDCG@5 sums `(2^grade - 1) / log2(rank + 1)` over the top 5 and divides by an ideal DCG. Two ideals are compared:

- **standard:** the best ordering of the chunks that this chunking produced (the textbook definition)
- **fixed ideal:** one whole-fact chunk at rank 1, the same target for every chunking

Same 12 questions, asked two ways, as Days 04 and 05:

| Chunking | Retriever | Original hit@3 / nDCG@5 / nDCG@5 fixed | Paraphrased hit@3 / nDCG@5 / nDCG@5 fixed |
| --- | --- | --- | --- |
| Sentence packing (160) | BM25 | 1.00 / 1.00 / 1.00 | 0.83 / 0.83 / 0.83 |
| Sentence packing (160) | char n-gram | 1.00 / 1.00 / 1.00 | 1.00 / 0.92 / 0.92 |
| Sentence packing (160) | RRF (BM25 + char) | 1.00 / 1.00 / 1.00 | 1.00 / 0.93 / 0.93 |
| Fixed size (120) | BM25 | 0.67 / **1.00** / 0.78 | 0.58 / 0.76 / 0.61 |
| Fixed size (120) | char n-gram | 0.67 / 0.94 / 0.74 | 0.50 / 0.76 / 0.58 |
| Fixed size (120) | RRF (BM25 + char) | 0.67 / 0.97 / 0.77 | 0.58 / 0.81 / 0.63 |

**Takeaway:** with fixed-size chunks, 4 of the 12 facts never sit whole in any chunk. Standard nDCG@5 still gave BM25 a perfect **1.00** on the original questions, because it only asks whether the retriever ordered *these* chunks well, and the best these chunks can offer is a half-fact. With a fixed ideal the same run scores 0.78, much closer to its hit@3 of 0.67. On sentence chunks both versions agree, since every fact exists whole. So nDCG is a good metric for comparing retrievers on one chunking, but comparing chunkings needs labels or an ideal that does not depend on the chunking. Graded labels also show partial credit that hit rate hides: on paraphrases, RRF on fixed chunks reaches 0.63 against a hit@3 of 0.58. With 12 questions one question moves a score by about 0.08, so this shows the mechanism, not a benchmark.

**Next:** the RAG v2 table (real question set, learned embeddings, reranker) reports nDCG@10 with a fixed ideal next to Recall@5 and MRR.
