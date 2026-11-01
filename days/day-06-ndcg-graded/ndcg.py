"""Day 06: nDCG@5 with graded relevance, next to the binary hit rate from Day 04.

Hit rate only asks "is the whole fact inside one top-k chunk?". A fixed-size chunk that holds
half of the fact scores zero there, although a reader (or an LLM) would still get part of the
answer. Graded labels make that visible: 2 = chunk holds the whole fact, 1 = chunk holds a
contiguous piece of at least half of it, 0 = otherwise. nDCG@5 then rewards high grades at
high ranks and divides by the best possible ordering for that question.

Run:  python ndcg.py
Test: python -m pytest -q
"""
import math
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day-05-hybrid-rrf"))
from hybrid import CORPUS, PARAPHRASED, QA, BM25, CharNgramIndex, HybridIndex, by_sentence, evaluate, fixed_size  # noqa: E402


def grade(chunk: str, fact: str) -> int:
    chunk = " ".join(chunk.split())
    if fact in chunk:
        return 2
    m = SequenceMatcher(None, fact, chunk, autojunk=False).find_longest_match(0, len(fact), 0, len(chunk))
    return 1 if m.size >= len(fact) / 2 else 0


def dcg(gains: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(rank + 1) for rank, g in enumerate(gains, 1))


def ndcg_at_k(index, chunks: list[str], question: str, fact: str, k: int = 5, fixed_ideal: bool = False) -> float:
    """Standard nDCG divides by the best ordering of THESE chunks. fixed_ideal=True divides by one
    whole-fact chunk at rank 1 instead (capped at 1.0), so a chunking that splits the fact is penalised."""
    got = [grade(chunks[i], fact) if score > 0 else 0 for i, score in index.search(question, k)]
    best = dcg([2]) if fixed_ideal else dcg(sorted((grade(c, fact) for c in chunks), reverse=True)[:k])
    return min(1.0, dcg(got) / best) if best else 0.0


def mean_ndcg(index, chunks: list[str], qa, k: int = 5, fixed_ideal: bool = False) -> float:
    return sum(ndcg_at_k(index, chunks, q, f, k, fixed_ideal) for q, f in qa) / len(qa)


if __name__ == "__main__":
    for name, chunks in [("fixed_size 120", fixed_size(CORPUS, 120)), ("sentence packing 160", by_sentence(CORPUS, 160))]:
        bm, ch = BM25(chunks), CharNgramIndex(chunks)
        for label, idx in [("bm25", bm), ("char-ngram", ch), ("rrf(bm25+char)", HybridIndex([bm, ch], len(chunks)))]:
            for qs_name, qs in [("original", QA), ("paraphrased", PARAPHRASED)]:
                hit3 = evaluate(idx, chunks, qs)["hit@3"]
                print(f"{name:<21} {label:<15} {qs_name:<12} hit@3={hit3:.2f}  ndcg@5={mean_ndcg(idx, chunks, qs):.2f}"
                      f"  ndcg@5_fixed_ideal={mean_ndcg(idx, chunks, qs, fixed_ideal=True):.2f}")
