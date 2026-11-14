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
| 05 | [Hybrid search with RRF](days/day-05-hybrid-rrf) | Fusing BM25 with a character n-gram retriever via RRF lifted paraphrase hit@3 from 0.83 to 1.00 and MRR from 0.83 to 0.90 on sentence chunks; on fixed-size chunks it only matched BM25 |
| 06 | [nDCG@5 with graded relevance](days/day-06-ndcg-graded) | Standard nDCG@5 gave BM25 on fixed-size chunks a perfect 1.00 although 4 of 12 facts never sat whole in a chunk (hit@3 0.67); with a fixed ideal it scored 0.78, so comparing chunkings needs an ideal that does not depend on the chunking |
| 07 | [Recall@k curves before a reranker](days/day-07-recall-depth) | No pool depth fixes two limits: fixed-size chunks cap recall at 0.67 (4 of 12 facts never whole), and BM25 gives 2 of 12 paraphrased answers a zero score so it stays at 0.83 at any depth; RRF with a character n-gram retriever reaches 1.00 by k=3 on sentence chunks |
| 08 | [Validating LLM JSON output](days/day-08-json-validation) | On 12 hand-written failure cases, strict json.loads parsed 6 but 4 of those were schema-invalid; tolerant extraction parsed 9 (fences, prose, trailing commas) without repairing truncation, and parse + schema check matched the expected verdict 12/12 |
| 09 | [Learned embeddings vs sparse retrievers](days/day-09-dense-vs-sparse) | bge-small alone matched character n-grams on paraphrases (hit@3 1.00, MRR 0.89 on sentence chunks) and lost one exact-wording question at rank 1; fused with BM25 via RRF it lifted paraphrase hit@1 from 0.83 to 0.92, and BM25 + char + dense gave the best MRR (1.00 original, 0.96 paraphrased) |
| 10 | [Cross-encoder reranking](days/day-10-cross-encoder-rerank) | A MiniLM cross-encoder on the fused top 3 moved the last paraphrased answer to rank 1 (hit@1 0.92 to 1.00) for about 30 ms per query on CPU; its negative logits first broke the harness's score > 0 hit rule (hit@1 0.33) until mapped through a sigmoid |
| 11 | ["Not found": abstention thresholds](days/day-11-abstain-threshold) | With 12 unanswerable questions added, the RRF score separated answerable from unanswerable poorly (AUROC 0.68; 42% of unanswerable still answered at its strictest useful cut-off), the cross-encoder probability much better (AUROC 0.85; at p >= 0.3, 71% answered right and 17% of unanswerable answered; at 0.9 none answered but 38% of real questions refused) |
| 12 | [Citation checks without a model](days/day-12-citation-check) | On 12 supported, 12 wrong-chunk and 9 fabricated citations, every check caught all wrong-chunk citations; content-word recall >= 0.75 let all 9 fabrications through, and "all content words inside one sentence" caught 8 of 9 while refusing 1 of 12 honest claims; a same-words role swap needs an entailment model or LLM judge |
| 13 | [NLI only where the cheap checks pass](days/day-13-nli-judge) | NLI alone (deberta-v3-xsmall, ONNX) accepted 2 entity swaps the one-sentence rule rejects (p 0.91, 0.95); the rule passed the role swap the model rejects (p 0.36); rule then NLI caught 9/9 fabrications and 12/12 wrong-chunk citations with 12 model calls of 33, about 25 ms each on CPU |
| 14 | [A real question set from real documents](days/day-14-real-docs) | On 16 user-style questions over six Python Language Reference sections, RRF (BM25 + char n-grams) on sentence chunks put the answer first for 0.25 and in the top 5 for 0.81 (toy corpus: hit@3 1.00); fixed 400-char chunks split 2 of 16 answers but had the best hit@1 (0.38), and two questions with no word overlap stayed out of every sparse top 10 |
| 15 | [Dense + cross-encoder on real docs](days/day-15-real-docs-dense-rerank) | On the Day 14 real question set (77 sentence chunks), adding bge-small to RRF raised Recall@5 0.81 -> 0.88; a MiniLM cross-encoder over the fused top 5 raised hit@1 0.25 -> 0.44 (about 0.5 s/query on CPU), while a pool of 20 cost 4x more and dropped Recall@5 to 0.75 |
| 16 | [Section-aware chunks with a title prefix](days/day-16-section-context) | Packing each Language Reference section on its own and starting every chunk with its title lifted lexical RRF hit@1 0.25 -> 0.44 and Recall@5 0.81 -> 0.88 on the real question set, matching the Day 15 cross-encoder's hit@1 without a model; BM25 alone lost one answer from its top 10 |

## Roadmap

| Week | Theme | Planned experiments |
| --- | --- | --- |
| 1 | Retrieval basics | chunking, BM25 from scratch, cosine search with NumPy, hit rate and MRR evaluation, hybrid search |
| 1b | RAG v2 (capstone for a LinkedIn post) | real question set from real documents, learned embeddings vs TF-IDF on paraphrases, hybrid BM25 + dense with RRF, cross-encoder reranker, nDCG@10 / MRR / Recall@5 table, then 4-bit LLM answers with citations (GPU step: ask Hussnain where to run it) |
| 2 | Reliable LLM output | prompt templates, JSON output validation, retry and backoff, token and cost estimator, response caching |
| 3 | Serving and automation | FastAPI RAG endpoint, streaming responses, webhook handler, n8n workflow template, scheduling jobs |
| 4 | Voice and agents | Whisper transcription script, tool calling loop, two agents with a handoff, simple agent eval, German vocabulary trainer CLI |
