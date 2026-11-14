"""Pure-Python helpers to compare two GRADIEND runs from their saved results JSON (no torch needed)."""
import re
from collections import Counter

LAYER = re.compile(r"(?:layer|h)\.(\d+)\.")


def topk_overlap(a, b):
    """|A & B| / min(|A|, |B|) for two lists of top-k parameter indices (same base model)."""
    a, b = set(a), set(b)
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def where(param_names):
    """Count top-k weights per block: 'layer N', 'embeddings' or 'head/other'."""
    out = Counter()
    for n in param_names:
        m = LAYER.search(n)
        out[f"layer {int(m.group(1)):02d}" if m else ("embeddings" if "embed" in n or ".wte" in n or ".wpe" in n else "head/other")] += 1
    return dict(sorted(out.items()))
