# Day 04: hit rate and MRR

**Question:** how do you measure a retriever instead of eyeballing single answers, and what do hit rate and MRR show that a single "found it / missed it" count hides?

`evaluate.py` adds a larger corpus (6 paragraphs about a RAG pipeline) and 12 questions, each asked two ways: with the document's own words (original) and mostly in other words (paraphrased). For every question it finds the 1-based rank of the first chunk that contains the whole answer fact, then computes:

- **hit@k**: share of questions whose answer chunk is in the top k
- **MRR@5**: mean of 1/rank (0 when the answer chunk is not in the top 5), so rank 1 counts 1.0 and rank 2 only 0.5

It reuses the Day 01 chunkers, the Day 02 BM25 and the Day 03 TF-IDF cosine index (`python evaluate.py`):

| Chunking | Questions | hit@1 | hit@3 | MRR@5 |
| --- | --- | --- | --- | --- |
| Fixed size (120) | original | 0.67 | 0.67 | 0.67 |
| Fixed size (120) | paraphrased | 0.50 | 0.58 | 0.53 |
| Fixed size + 30 overlap | original | 0.83 | 0.83 | 0.83 |
| Fixed size + 30 overlap | paraphrased | 0.42 | 0.67 | 0.54 |
| Sentence packing (max 160) | original | 1.00 | 1.00 | 1.00 |
| Sentence packing (max 160) | paraphrased | 0.83 | 0.83 | 0.83 |

BM25 and cosine produced identical numbers in every row, so the table shows one value per row.

**Takeaway:** on paraphrased questions, overlap raised hit@3 (0.58 to 0.67) but lowered hit@1 (0.50 to 0.42): in each of the three rank-2 cases a neighbouring overlap chunk ranked first, and in two of them it held only part of the fact. MRR moved only from 0.53 to 0.54, which is why it is worth reporting next to hit rate: hit@3 alone would call overlap a clear win. Sentence packing was best on both question sets, and paraphrasing cost it 2 of 12 questions. With only 12 questions, one question moves a score by 0.08, so this is a toy set that shows how the metrics behave, not a benchmark.

**Next:** day 05 tries hybrid search with reciprocal rank fusion, measured with this harness. Because BM25 and TF-IDF cosine rank almost the same here, fusing them alone is unlikely to help; the paraphrased set is where a learned-embedding retriever should make the difference.
