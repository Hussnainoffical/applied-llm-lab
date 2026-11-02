# Day 07: Recall@k curves before a reranker

**Question:** a reranker can only reorder the candidates the first stage hands it. How deep does the candidate pool have to be before every answerable fact is in it, and what can no pool depth fix?

`recall_depth.py` computes Recall@k for k = 1 up to the number of chunks: the share of questions whose whole-fact chunk is ranked within the top k (a chunk with score 0 is not a candidate). It also reports the **ceiling** (share of facts that sit whole in at least one chunk, the best any retriever could do) and the **saturation depth** (smallest k at which the curve stops growing). Same 12 questions, asked two ways, as Days 04 to 06.

| Chunking (ceiling) | Retriever | Original R@1 / R@3 / R@all | Paraphrased R@1 / R@3 / R@5 / R@all | Paraphrased saturates at |
| --- | --- | --- | --- | --- |
| Sentence packing 160, 8 chunks (1.00) | BM25 | 1.00 / 1.00 / 1.00 | 0.83 / 0.83 / 0.83 / **0.83** | k=1 |
| Sentence packing 160, 8 chunks (1.00) | char n-gram | 1.00 / 1.00 / 1.00 | 0.83 / 1.00 / 1.00 / 1.00 | k=3 |
| Sentence packing 160, 8 chunks (1.00) | RRF (BM25 + char) | 1.00 / 1.00 / 1.00 | 0.83 / 1.00 / 1.00 / 1.00 | k=3 |
| Fixed size 120, 9 chunks (**0.67**) | BM25 | 0.67 / 0.67 / 0.67 | 0.50 / 0.58 / 0.58 / 0.58 | k=3 |
| Fixed size 120, 9 chunks (0.67) | char n-gram | 0.58 / 0.67 / 0.67 | 0.42 / 0.50 / 0.58 / 0.67 | k=8 |
| Fixed size 120, 9 chunks (0.67) | RRF (BM25 + char) | 0.67 / 0.67 / 0.67 | 0.50 / 0.58 / 0.58 / 0.67 | k=8 |

**Takeaway:** two different limits show up, and a deeper pool fixes neither. (1) **Chunking ceiling:** with fixed-size chunks 4 of 12 facts never sit whole in any chunk, so recall stops at 0.67 at every depth; a reranker on top could never exceed that. (2) **Lexical gap:** on paraphrases BM25 gives 2 of 12 right chunks a score of 0, so they never enter the pool, and BM25 stays at 0.83 even when it returns all 8 chunks. Adding the character n-gram retriever with RRF closes that gap by k=3 on sentence chunks. On fixed chunks the fused pool only reaches its ceiling at k=8 of 9 chunks, so a reranker would need almost the whole corpus as input. Lesson for a real system: plot Recall@k of the first stage up to the planned reranker depth (often 50 to 150) and check it against the ceiling before tuning the reranker. With 8 to 9 chunks and 12 questions this shows the mechanism, not realistic pool sizes.

**Next:** RAG v2 with a real question set, where the corpus is large enough for pool depths of 20 to 100 to matter.
