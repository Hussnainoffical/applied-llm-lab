"""Day 14: a real question set from real documents.

Days 01-13 ran on a hand-written toy corpus. Here the corpus is six sections of the Python Language
Reference that ship with every CPython install (pydoc_data.topics: try, raise, for, while, import,
with), about 25k characters with grammar rules, examples and version notes. 16 questions are written
the way a user asks them, not copied from the text; each has one exact answer span from the docs.

Run:  python real_docs.py
Test: python -m pytest -q
"""
import sys
from pathlib import Path

from pydoc_data.topics import topics

for day in ("day-04-hit-rate-mrr", "day-05-hybrid-rrf"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / day))
from evaluate import answer_rank  # noqa: E402
from hybrid import BM25, CharNgramIndex, HybridIndex, by_sentence, fixed_size  # noqa: E402

SECTIONS = ("try", "raise", "for", "while", "import", "with")
CORPUS = "\n\n".join(topics[s] for s in SECTIONS)

QA = [
    ("Does a bare except have to be the final handler?", 'An expression-less "except" clause, if present, must be last'),
    ("Why is my exception variable gone after the handler finishes?", 'it is cleared at the end of the "except" clause'),
    ("What would the caught exception keep alive if Python did not delete the name?", "they form a reference cycle with the stack frame"),
    ("If finally runs return, what happens to the error that was in flight?", "the saved exception is discarded"),
    ("When does the else block after try actually run?", 'no exception was raised, and no "return", "continue", or "break" statement was executed'),
    ("What does except star handle?", "specify one or more handlers for groups of exceptions"),
    ("What does a raise with nothing after it do?", "re-raises the exception that is currently being handled"),
    ("Where does the original error end up when I chain with from?", 'attached to the raised exception as the "__cause__" attribute'),
    ("How do I hide the context of a chained exception?", '"None" is now permitted as "Y" in "raise X from Y"'),
    ("When does the else of a for loop execute?", 'When the iterator is exhausted, the suite in the "else" clause, if present, is executed'),
    ("Does break in a while loop skip its else part?", 'terminates the loop without executing the "else" clause'),
    ("Which names does star import pull in from a module?", "namespace for a variable named \"__all__\""),
    ("Can I import a sibling module without writing the package name?", "make a relative import within the same top package"),
    ("What is a future statement for?", "a directive to the compiler that a particular module should be compiled"),
    ("Can one with statement manage several context managers at once?", "Support for multiple context expressions"),
    ("Is __exit__ guaranteed to run if __enter__ succeeded?", 'returns without an error, then "__exit__()" will always be called'),
]


def ceiling(chunks: list[str], qa=QA) -> float:
    """Share of answer spans that sit whole in at least one chunk: the best any retriever can do."""
    flat = [" ".join(c.split()) for c in chunks]
    return sum(any(f in c for c in flat) for _, f in qa) / len(qa)


def scores(index, chunks: list[str], qa=QA, k: int = 10) -> dict:
    ranks = [answer_rank(index, chunks, q, f, k) for q, f in qa]
    return {"hit@1": sum(r == 1 for r in ranks) / len(qa),
            "recall@5": sum(r is not None and r <= 5 for r in ranks) / len(qa),
            "mrr@10": sum(1 / r for r in ranks if r) / len(qa),
            "missed": [q for (q, _), r in zip(qa, ranks) if r is None]}


if __name__ == "__main__":
    for name, chunks in [("fixed_size 400", fixed_size(CORPUS, 400)), ("sentence packing 400", by_sentence(CORPUS, 400))]:
        bm, ch = BM25(chunks), CharNgramIndex(chunks)
        print(f"{name}: {len(chunks)} chunks, ceiling {ceiling(chunks):.2f}")
        for label, idx in [("bm25", bm), ("char-ngram", ch), ("rrf(bm25+char)", HybridIndex([bm, ch], len(chunks)))]:
            s = scores(idx, chunks)
            print(f"  {label:<15} hit@1={s['hit@1']:.2f} R@5={s['recall@5']:.2f} MRR@10={s['mrr@10']:.2f} missed={len(s['missed'])}")
