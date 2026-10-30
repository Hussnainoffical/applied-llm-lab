"""Day 04: hit rate and MRR for BM25 and TF-IDF cosine over 12 questions, asked two ways.

Run:  python evaluate.py
Test: python -m pytest -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day-03-cosine"))
from cosine import BM25, CosineIndex, by_sentence, fixed_size  # noqa: E402

CORPUS = (
    "A RAG pipeline has two phases. The indexing phase runs once per document set. "
    "The query phase runs for every user question.\n\n"
    "During indexing, PDF files are converted to plain text before chunking. Tables often lose "
    "their row structure in this step. Each chunk keeps the file name and page number as metadata.\n\n"
    "Embeddings map each chunk to a dense vector. A small model like all-MiniLM-L6-v2 produces "
    "384 dimensions. Vectors are stored in an index such as FAISS for nearest-neighbour search.\n\n"
    "Lexical search with BM25 rewards exact term matches. It works well for product codes and names. "
    "Dense search handles synonyms but can miss rare identifiers.\n\n"
    "Hybrid search merges both ranked lists. Reciprocal rank fusion adds one over sixty plus the rank "
    "from each list. A cross-encoder reranker then rescores the top candidates jointly with the query.\n\n"
    "Evaluation needs a labelled question set. Hit rate counts questions whose answer chunk appears in "
    "the top k. Mean reciprocal rank also rewards putting that chunk near the top."
)
# (question, fact that must sit whole inside a chunk for it to count as the answer chunk)
QA = [
    ("how many phases does a RAG pipeline have", "A RAG pipeline has two phases"),
    ("when does the query phase run", "The query phase runs for every user question"),
    ("what happens to PDF files during indexing", "PDF files are converted to plain text before chunking"),
    ("what do tables lose", "Tables often lose their row structure in this step"),
    ("which metadata does each chunk keep", "Each chunk keeps the file name and page number as metadata"),
    ("how many dimensions does all-MiniLM-L6-v2 produce", "produces 384 dimensions"),
    ("where are vectors stored", "Vectors are stored in an index such as FAISS"),
    ("what does BM25 reward", "Lexical search with BM25 rewards exact term matches"),
    ("what can dense search miss", "Dense search handles synonyms but can miss rare identifiers"),
    ("what does reciprocal rank fusion add", "Reciprocal rank fusion adds one over sixty plus the rank"),
    ("what does a cross-encoder reranker do", "A cross-encoder reranker then rescores the top candidates"),
    ("what does mean reciprocal rank reward", "also rewards putting that chunk near the top"),
]

PARAPHRASED = [  # same facts, asked mostly in other words
    ("into how many stages is retrieval-augmented generation split", QA[0][1]),
    ("what runs each time someone asks something", QA[1][1]),
    ("how are PDFs handled before splitting", QA[2][1]),
    ("which structure gets broken for tabular data", QA[3][1]),
    ("what information about the source file is attached to every piece", QA[4][1]),
    ("how large is the vector from the MiniLM model", QA[5][1]),
    ("which library holds the vectors for nearest neighbour lookup", QA[6][1]),
    ("which method is good at exact keywords", QA[7][1]),
    ("what is the weakness of embedding search with rare ids", QA[8][1]),
    ("how does RRF combine rankings", QA[9][1]),
    ("what model rescores candidates together with the question", QA[10][1]),
    ("which metric cares about the position of the answer chunk", QA[11][1]),
]


def answer_rank(index, chunks: list[str], question: str, fact: str, k: int = 5) -> int | None:
    """1-based rank of the first top-k chunk that holds the whole fact with a non-zero score."""
    flat = [" ".join(c.split()) for c in chunks]
    for rank, (i, score) in enumerate(index.search(question, k), 1):
        if score > 0 and fact in flat[i]:
            return rank
    return None


def evaluate(index, chunks: list[str], qa=QA, k: int = 5) -> dict:
    ranks = [answer_rank(index, chunks, q, f, k) for q, f in qa]
    return {"hit@1": sum(r == 1 for r in ranks) / len(qa), "hit@3": sum(r is not None and r <= 3 for r in ranks) / len(qa),
            "mrr@5": sum(1 / r for r in ranks if r) / len(qa)}


if __name__ == "__main__":
    for name, chunks in [("fixed_size 120", fixed_size(CORPUS, 120)),
                         ("fixed 120 + 30 overlap", fixed_size(CORPUS, 120, 30)),
                         ("sentence packing 160", by_sentence(CORPUS, 160))]:
        for label, idx in [("bm25", BM25(chunks)), ("cosine", CosineIndex(chunks))]:
            for qs_name, qs in [("original", QA), ("paraphrased", PARAPHRASED)]:
                m = evaluate(idx, chunks, qs)
                print(f"{name:<24} {label:<7} {qs_name:<12}" + "  ".join(f"{k}={v:.2f}" for k, v in m.items()))
