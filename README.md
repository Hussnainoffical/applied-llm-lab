# applied-llm-lab

One small, working experiment a day on applied LLMs, RAG and AI automation. Each day is a self-contained folder with runnable code, tests and a short note on what I learned.

By [Muhammad Hussnain Jamil](https://www.linkedin.com/in/hussnain-jamil) · [Portfolio](https://muhammad-hussnain-jamil-portfolio.vercel.app)

## Run any day

```
cd days/day-01-chunking
pip install pytest numpy
python -m pytest -q
python chunking.py
```

## Log

| Day | Topic | Takeaway |
| --- | --- | --- |
| 01 | [Chunking strategies for RAG](days/day-01-chunking) | Fixed-size chunks cut sentences; sentence packing kept every sentence whole at a similar chunk size |
| 02 | [BM25 retrieval from scratch](days/day-02-bm25) | On 5 questions, sentence-packed chunks put the whole fact in BM25's top result 5/5 times vs 2/5 for fixed-size chunks (4/5 with overlap) |
| 03 | [Cosine search with NumPy](days/day-03-cosine) | TF-IDF cosine had the same hit counts as BM25 (5/5 with sentence chunks); on 3 paraphrased questions both scored 0/3, so paraphrases need learned embeddings |
| 04 | [Hit rate and MRR](days/day-04-hit-rate-mrr) | 12 questions asked two ways: on paraphrases, 30-char overlap raised hit@3 (0.58 to 0.67) but lowered hit@1 (0.50 to 0.42), and MRR barely moved (0.53 to 0.54); sentence packing scored 1.00 on original and 0.83 on paraphrased questions |

## Roadmap

| Week | Theme | Planned experiments |
| --- | --- | --- |
| 1 | Retrieval basics | chunking, BM25 from scratch, cosine search with NumPy, hit rate and MRR evaluation, hybrid search |
| 1b | RAG v2 (capstone for a LinkedIn post) | real question set from real documents, learned embeddings vs TF-IDF on paraphrases, hybrid BM25 + dense with RRF, cross-encoder reranker, nDCG@10 / MRR / Recall@5 table, then 4-bit LLM answers with citations (GPU step: ask Hussnain where to run it) |
| 2 | Reliable LLM output | prompt templates, JSON output validation, retry and backoff, token and cost estimator, response caching |
| 3 | Serving and automation | FastAPI RAG endpoint, streaming responses, webhook handler, n8n workflow template, scheduling jobs |
| 4 | Voice and agents | Whisper transcription script, tool calling loop, two agents with a handoff, simple agent eval, German vocabulary trainer CLI |
