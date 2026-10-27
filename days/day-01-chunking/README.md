# Day 01: Chunking strategies for RAG

**Question:** how much does the chunking method change what a retriever gets to see?

Three methods on the same sample text (`python chunking.py`):

| Method | Chunks | Avg chars | Chunks cut mid-sentence |
| --- | --- | --- | --- |
| Fixed size (120) | 4 | 107 | 3 |
| Fixed size + 30 overlap | 5 | 109 | 4 |
| Sentence packing (max 160) | 4 | 105 | 0 |
| Paragraph | 3 | 141 | 0 |

**Takeaway:** at about the same chunk size, sentence packing kept every sentence whole while fixed-size chunking cut 3 of 4 chunks mid-sentence. Overlap repeats text but doesn't fix the cuts. Paragraph splitting is best when the document already has structure.

**Next:** day 02 builds BM25 retrieval from scratch to measure how these chunks actually rank.
