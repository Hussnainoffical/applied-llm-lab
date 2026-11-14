"""Summarise results/*.json: encoder correlation, rewrite effect on probe sentences, top-k overlap between runs."""
import json
from itertools import combinations
from pathlib import Path

from overlap import topk_overlap, where

R = {p.stem: json.loads(p.read_text()) for p in sorted(Path("results").glob("*.json"))}

print("| Run | Model | Train pairs | Encoder corr. | Train time |")
print("|---|---|---|---|---|")
for n, r in R.items():
    c = r["encoder_correlation"]
    print(f"| {n} | {r['model']} | {r['n_train']} | {c:.3f} | {r['train_seconds']} s |" if isinstance(c, float) else f"| {n} | {r['model']} | {r['n_train']} | {c} | {r['train_seconds']} s |")

print("\nRewrite (strengthen", ", ".join(f"{n}:{r['rewrite_class']}" for n, r in R.items()), ")")
for n, r in R.items():
    for s, before in r["probe_before"].items():
        after = r["probe_after"][s]
        print(f"  {n:16s} {s[:48]:48s} " + "  ".join(f"{w} {before[w]:.3f}->{after[w]:.3f}" for w in before))

print("\nTop-1000 decoder weights: overlap |A&B|/1000 (same base model only)")
for a, b in combinations(R, 2):
    if len(R[a]["topk_index"]) and len(R[b]["topk_index"]) and R[a]["model"].split("-")[0] != "distilgpt2" and R[b]["model"] != "distilgpt2":
        print(f"  {a} vs {b}: {topk_overlap(R[a]['topk_index'], R[b]['topk_index']):.2f}")

print("\nWhere the top-1000 sit")
for n, r in R.items():
    print(f"  {n}: {where(r['topk_param'])}")
