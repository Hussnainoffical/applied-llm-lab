"""Day 12: does the cited chunk actually support the sentence? Deterministic citation checks.

RAG v2, step 4. An answer sentence that ends in [chunk 3] looks grounded, but the citation can point
to the wrong chunk, or the chunk can say something else (another number, another method). Four
checks run without a model: exact containment, content-word recall, recall plus a rule that every
number or identifier must appear in the chunk, and "all content words inside one sentence".

Run:  python citation_check.py
Test: python -m pytest -q
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day-04-hit-rate-mrr"))
from evaluate import CORPUS, by_sentence  # noqa: E402
from bm25 import tokenize  # noqa: E402  (day-02, on the path via day-04)

STOP = set("a an the of to in for and or is are be by with as it its this that each then also such".split())
CHUNKS = by_sentence(CORPUS, 160)

# (claim as an LLM might phrase it, chunk it should cite)
SUPPORTED = [
    ("The RAG pipeline has two phases.", 0), ("The query phase runs for every user question.", 0),
    ("PDF files are converted to plain text before chunking.", 1), ("Tables often lose their row structure.", 1),
    ("Each chunk keeps its file name and page number as metadata.", 2), ("all-MiniLM-L6-v2 produces 384 dimensions.", 3),
    ("Vectors are stored in an index such as FAISS.", 3), ("BM25 rewards exact term matches.", 4),
    ("Dense search can miss rare identifiers.", 4), ("Reciprocal rank fusion adds one over sixty plus the rank.", 5),
    ("A cross-encoder reranker rescores the top candidates with the query.", 6), ("Mean reciprocal rank rewards putting the answer chunk near the top.", 7),
]
WRONG_CHUNK = [(claim, (c + 3) % len(CHUNKS)) for claim, c in SUPPORTED]  # right sentence, wrong citation
FABRICATED = [  # cites the topically right chunk, but the chunk does not say this
    ("The RAG pipeline has three phases.", 0), ("all-MiniLM-L6-v2 produces 768 dimensions.", 3),
    ("Reciprocal rank fusion adds one over one hundred plus the rank.", 5), ("Vectors are stored in an index such as Pinecone.", 3),
    ("Each chunk keeps the author name and page number as metadata.", 2), ("Dense search rewards exact term matches.", 4),
    ("BM25 can miss rare identifiers.", 4), ("Tables never lose their row structure.", 1),
    ("Dense search handles rare identifiers but can miss synonyms.", 4),  # swapped roles, same words
]


def words(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOP]


def exact(claim: str, chunk: str) -> bool:
    return claim.rstrip(".").lower() in " ".join(chunk.split()).lower()


def recall(claim: str, chunk: str) -> float:
    w, c = words(claim), set(words(chunk))
    return sum(t in c for t in w) / max(len(w), 1)


def overlap(claim: str, chunk: str, t: float = 0.75) -> bool:
    return recall(claim, chunk) >= t


def overlap_ids(claim: str, chunk: str, t: float = 0.75) -> bool:
    ids = [x for x in re.findall(r"[A-Za-z0-9-]+", claim) if re.search(r"\d", x) or x[:1].isupper() and x != claim.split()[0]]
    return overlap(claim, chunk, t) and all(x.lower() in chunk.lower() for x in ids)


def same_sentence(claim: str, chunk: str) -> bool:
    """Every content word of the claim must appear inside ONE sentence of the cited chunk."""
    return any(set(words(claim)) <= set(words(s)) for s in re.split(r"(?<=[.!?])\s+", chunk))


def score(check) -> dict:
    ok = sum(check(c, CHUNKS[i]) for c, i in SUPPORTED) / len(SUPPORTED)
    caught_wrong = sum(not check(c, CHUNKS[i]) for c, i in WRONG_CHUNK) / len(WRONG_CHUNK)
    caught_fab = sum(not check(c, CHUNKS[i]) for c, i in FABRICATED) / len(FABRICATED)
    return {"supported accepted": ok, "wrong chunk caught": caught_wrong, "fabricated caught": caught_fab}


if __name__ == "__main__":
    for name, check in [("exact substring", exact), ("word recall >= 0.75", overlap), ("recall + ids/numbers", overlap_ids),
                        ("all words, one sentence", same_sentence)]:
        print(f"{name:<25}" + "  ".join(f"{k}={v:.2f}" for k, v in score(check).items()))
    print("one-sentence rule: rejected supported", [c for c, i in SUPPORTED if not same_sentence(c, CHUNKS[i])],
          "| passed fabricated", [c for c, i in FABRICATED if same_sentence(c, CHUNKS[i])])
