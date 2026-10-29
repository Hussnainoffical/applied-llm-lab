# Day 03: cosine search with NumPy

**Question:** does vector search with cosine similarity find different chunks than BM25 (Day 02), and what happens when a question uses other words than the document?

`cosine.py` builds TF-IDF vectors with plain NumPy (same tokenizer as Day 02, smoothed IDF), normalises every row to length 1, and answers a query with one matrix-vector product. It runs on the Day 01 chunkings with the 5 Day 02 questions and 3 paraphrased questions that ask for the same facts in different words (`python cosine.py`):

| Chunking (from Day 01) | Original Qs: BM25 | Original Qs: cosine | Paraphrased Qs: BM25 | Paraphrased Qs: cosine |
| --- | --- | --- | --- | --- |
| Fixed size (120) | 2 / 5 | 2 / 5 | 0 / 3 | 1 / 3 |
| Fixed size + 30 overlap | 4 / 5 | 4 / 5 | 0 / 3 | 0 / 3 |
| Sentence packing (max 160) | 5 / 5 | 5 / 5 | 0 / 3 | 0 / 3 |

A hit means a non-zero score and the whole fact inside the top-ranked chunk.

**Takeaway:** on this small corpus TF-IDF cosine and BM25 picked the same top chunk in 14 of 15 question and chunking pairs (the one difference was a miss for both) and had identical hit counts, so chunking mattered more than the scoring formula. With paraphrased questions both failed. The single cosine hit is an accident: the fixed-size chunker cut "con|text", and the leftover "text" matched the word "text" in the question. Word-count vectors only match words, so paraphrases need learned embeddings. This is a toy set that shows the mechanism, not a benchmark.

**Next:** day 04 measures hit rate and MRR over a larger question set.
