"""Day 16: section-aware chunks with a title prefix (cheap contextual retrieval).

Day 14 packed sentences across the whole corpus, so a chunk can start in one statement's section and
end in the next, and most chunks never say which statement they describe ("the else clause" could be
try, for or while). Here each Language Reference section is chunked on its own and every chunk gets the
section title as a short prefix, then the same lexical retrievers and 16 questions are scored again.

Run:  python section_context.py
Test: python -m pytest -q
"""
import sys
from pathlib import Path

from pydoc_data.topics import topics

for day in ("day-04-hit-rate-mrr", "day-05-hybrid-rrf", "day-14-real-docs"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / day))
from hybrid import BM25, CharNgramIndex, HybridIndex, by_sentence  # noqa: E402
from real_docs import CORPUS, QA, SECTIONS, ceiling, scores  # noqa: E402


def per_section(size: int = 400, prefix: bool = False) -> list[str]:
    """Sentence-pack each section separately; optionally start every chunk with its section title."""
    chunks = []
    for s in SECTIONS:
        title = f'The "{s}" statement. '
        chunks += [title + c if prefix else c for c in by_sentence(topics[s], size)]
    return chunks


def retrievers(chunks: list[str]) -> dict:
    bm, ch = BM25(chunks), CharNgramIndex(chunks)
    return {"bm25": bm, "char": ch, "rrf(bm25+char)": HybridIndex([bm, ch], len(chunks))}


if __name__ == "__main__":
    setups = {"whole corpus (Day 14)": by_sentence(CORPUS, 400),
              "per section": per_section(400),
              "per section + title": per_section(400, prefix=True)}
    for name, chunks in setups.items():
        print(f"{name}: {len(chunks)} chunks, ceiling {ceiling(chunks):.2f}")
        for r, idx in retrievers(chunks).items():
            s = scores(idx, chunks)
            print(f"  {r:<15} hit@1={s['hit@1']:.2f} R@5={s['recall@5']:.2f} MRR@10={s['mrr@10']:.2f} missed={len(s['missed'])}")
