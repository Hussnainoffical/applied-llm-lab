# Day 09: Learned embeddings vs sparse retrievers on paraphrases

**Question:** BM25 and TF-IDF scored 0 on paraphrases that share no words with the answer (Day 03), and character n-grams only rescued part of them (Day 05). Does a small learned embedding model close that gap, and does it still help once it is fused with BM25?

This is the first step of the RAG v2 track (roadmap row 1b). `dense.py` adds `DenseIndex` (chunks embedded once, L2-normalised, cosine by one matrix-vector product) with **BAAI/bge-small-en-v1.5** (384 dims, ONNX on CPU via fastembed; queries go through `query_embed`, which adds the model's retrieval instruction). It runs through the unchanged Day 04 harness on the same corpus, chunkings and 12 + 12 questions, alone and fused with RRF (k = 60).

Sentence packing (160 chars, 8 chunks):

| Retriever | Original hit@1 / hit@3 / MRR@5 | Paraphrased hit@1 / hit@3 / MRR@5 |
| --- | --- | --- |
| BM25 | 1.00 / 1.00 / 1.00 | 0.83 / 0.83 / 0.83 |
| char n-gram | 1.00 / 1.00 / 1.00 | 0.83 / 1.00 / 0.89 |
| dense (bge-small) | 0.92 / 1.00 / 0.96 | 0.83 / 1.00 / 0.89 |
| RRF(BM25 + dense) | 0.92 / 1.00 / 0.96 | 0.92 / 1.00 / 0.94 |
| RRF(BM25 + char + dense) | 1.00 / 1.00 / 1.00 | 0.92 / 1.00 / 0.96 |

Fixed-size chunks (120 chars, 9 chunks) stay capped at 0.67 on original questions for every retriever, because 4 of 12 facts never sit whole in a chunk (Day 07 ceiling); the best paraphrase hit@3 there is 0.67 (RRF of BM25 + dense).

**Takeaway:** on this tiny corpus the learned model alone does not beat the character n-gram retriever on paraphrases (same 1.00 hit@3, same 0.89 MRR) and it loses one exact-wording question at rank 1 that BM25 gets right. Its value shows in fusion: BM25 + dense lifts paraphrase hit@1 from 0.83 to 0.92, and fusing all three gives the best MRR on both question sets. Chunking still matters more than the retriever: no model recovers a fact that was cut in half. With 8 chunks and 12 questions one question moves a metric by 0.08, so this shows the mechanism, not a ranking of models.

**Next (RAG v2):** a larger question set from real documents, so the differences are measured on more than a handful of chunks, then a cross-encoder reranker on the fused top k.
