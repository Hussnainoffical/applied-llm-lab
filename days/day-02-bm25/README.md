# Day 02: BM25 retrieval from scratch

**Question:** once the Day 01 chunks are indexed, does the top-ranked chunk contain the whole fact a question needs?

`bm25.py` implements Okapi BM25 with no libraries: tokenizer with a small stopword list, IDF, term-frequency saturation (`k1 = 1.5`) and length normalisation (`b = 0.75`). It ranks the chunks from the Day 01 sample text for 5 questions, each paired with the exact fact the top chunk must contain in full (`python bm25.py`):

| Chunking (from Day 01) | Chunks | Whole fact in top-1 result |
| --- | --- | --- |
| Fixed size (120) | 4 | 2 / 5 |
| Fixed size + 30 overlap | 5 | 4 / 5 |
| Sentence packing (max 160) | 4 | 5 / 5 |

**Takeaway:** with the same retriever, sentence packing delivered the complete fact in the top result for every question, fixed-size chunks only for 2 of 5. Overlap recovered most of the gap at the cost of one extra chunk to index. This is a 5-question toy set, so it shows the mechanism, not a benchmark.

**Next:** day 03 adds cosine search over NumPy vectors to compare with BM25.
