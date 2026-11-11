"""Day 13: an NLI model, but only for claims that pass the cheap checks (RAG v2, step 5).

Day 12 showed that a bag-of-words rule ("all content words inside one sentence of the cited chunk")
catches every wrong-chunk citation and 8 of 9 fabrications, but not a role swap with the same words.
Here a small NLI cross-encoder (Xenova/nli-deberta-v3-xsmall, ONNX on CPU) reads (chunk, claim) and
returns P(entailment). Measured three ways: NLI alone, the one-sentence rule alone, and the cascade
(rule first, model only for claims that pass), with the number of model calls and time per call.

Run:  python nli_check.py   (downloads the ONNX model on first use)
Test: python -m pytest -q   (no model needed)
"""
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day-12-citation-check"))
from citation_check import CHUNKS, FABRICATED, SUPPORTED, WRONG_CHUNK, same_sentence  # noqa: E402

Entail = Callable[[str, str], float]  # (premise = cited chunk, hypothesis = claim) -> P(entailment)


def nli_model(repo: str = "Xenova/nli-deberta-v3-xsmall") -> Entail:
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
    tok.enable_truncation(512)
    labels = json.load(open(hf_hub_download(repo, "config.json")))["id2label"]
    ent = int(next(k for k, v in labels.items() if v.lower() == "entailment"))  # label order differs per model
    sess = ort.InferenceSession(hf_hub_download(repo, "onnx/model.onnx"))
    names = {i.name for i in sess.get_inputs()}

    def entail(premise: str, hypothesis: str) -> float:
        e = tok.encode(premise, hypothesis)
        feed = {"input_ids": e.ids, "attention_mask": e.attention_mask, "token_type_ids": e.type_ids}
        logits = sess.run(None, {k: np.array([v], dtype=np.int64) for k, v in feed.items() if k in names})[0][0]
        z = np.exp(logits - logits.max())
        return float(z[ent] / z.sum())

    return entail


def cascade(entail: Entail, t: float = 0.5, calls: list | None = None) -> Callable[[str, str], bool]:
    """One-sentence rule first; only claims that pass it cost a model call."""
    def check(claim: str, chunk: str) -> bool:
        if not same_sentence(claim, chunk):
            return False
        if calls is not None:
            calls.append(claim)
        return entail(chunk, claim) >= t
    return check


def score(check) -> dict:
    rate = lambda pairs, want: sum(check(c, CHUNKS[i]) == want for c, i in pairs) / len(pairs)  # noqa: E731
    return {"supported accepted": rate(SUPPORTED, True), "wrong chunk caught": rate(WRONG_CHUNK, False),
            "fabricated caught": rate(FABRICATED, False)}


if __name__ == "__main__":
    model, seconds, calls = nli_model(), [], []

    def timed(premise: str, hypothesis: str) -> float:
        t = time.perf_counter()
        p = model(premise, hypothesis)
        seconds.append(time.perf_counter() - t)
        return p

    for name, check in [("one-sentence rule", same_sentence), ("NLI alone, p>=0.5", lambda c, ch: timed(ch, c) >= 0.5),
                        ("rule -> NLI, p>=0.5", cascade(timed, 0.5, calls))]:
        print(f"{name:<22}" + "  ".join(f"{k}={v:.2f}" for k, v in score(check).items()))
    print(f"cascade model calls: {len(calls)} of {len(SUPPORTED) + len(WRONG_CHUNK) + len(FABRICATED)}; "
          f"{1000 * sum(seconds) / len(seconds):.0f} ms per call on CPU")
    for c, i in SUPPORTED + FABRICATED:
        print(f"  p(entail)={model(CHUNKS[i], c):.2f}  rule={'pass' if same_sentence(c, CHUNKS[i]) else 'fail'}  {c}")
