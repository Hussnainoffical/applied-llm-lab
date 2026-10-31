# Day 05: hybrid search with reciprocal rank fusion

**Question:** does fusing two retrievers with RRF help, and when does it not?

Day 04 showed that BM25 and word-level TF-IDF cosine rank this corpus identically, so fusing those two would add nothing. `hybrid.py` pairs BM25 with a retriever that fails differently: TF-IDF cosine over **character 3- to 5-grams** (words padded with spaces, so word starts and ends are marked). It can match `PDFs` to `PDF` or `rankings` to `rank`, which a word tokenizer without stemming cannot.

**Reciprocal rank fusion** ignores raw scores and adds `1 / (60 + rank)` from every list a chunk appears in. Chunks that a retriever scored 0 are left out of that retriever's list, so fusion cannot invent matches. Measured with the Day 04 harness (12 questions, asked two ways):

| Chunking | Retriever | Original hit@1 / hit@3 / MRR@5 | Paraphrased hit@1 / hit@3 / MRR@5 |
| --- | --- | --- | --- |
| Sentence packing (160) | BM25 | 1.00 / 1.00 / 1.00 | 0.83 / 0.83 / 0.83 |
| Sentence packing (160) | char n-gram | 1.00 / 1.00 / 1.00 | 0.83 / 1.00 / 0.89 |
| Sentence packing (160) | RRF (BM25 + char) | 1.00 / 1.00 / 1.00 | 0.83 / 1.00 / 0.90 |
| Fixed size (120) | BM25 | 0.67 / 0.67 / 0.67 | 0.50 / 0.58 / 0.53 |
| Fixed size (120) | char n-gram | 0.58 / 0.67 / 0.62 | 0.42 / 0.50 / 0.47 |
| Fixed size (120) | RRF (BM25 + char) | 0.67 / 0.67 / 0.67 | 0.50 / 0.58 / 0.53 |

**Takeaway:** with sentence chunks, fusion found both paraphrased questions BM25 missed completely ("how does RRF combine rankings" went from not found to rank 1, the "stages" question to rank 3), so hit@3 rose from 0.83 to 1.00 and MRR from 0.83 to 0.90. The cost: one question BM25 had at rank 1 slipped to rank 2, so hit@1 stayed at 0.83. With fixed-size chunks, fusion only matched BM25: the n-gram retriever was weaker there and RRF cannot recover an answer chunk that was cut in half. Fusion helps when the second retriever makes different mistakes, and it does not fix bad chunking. With 12 questions one question moves a score by 0.08, so this shows the mechanism, not a benchmark.

**Next:** replace the n-gram retriever with a learned embedding model on a real question set (RAG v2 roadmap).
