"""Day 18: what is actually inside a Q4_0, Q4_K_M and Q8_0 GGUF file.

Reads the tensor table of each file (no inference) and reports bits per weight,
the quantization type per weight role, and which layers a "mixed" K-quant stores
in a different type.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROLES = ["token_embd", "output", "attn_q", "attn_k", "attn_v", "attn_output",
         "ffn_gate", "ffn_up", "ffn_down", "norm"]
QK_K = 256  # K-quant super-block size


def role(name: str) -> str:
    """Map a GGUF tensor name (e.g. blk.3.attn_v.weight) to its role."""
    if "norm" in name:
        return "norm"
    for r in ROLES:
        if re.search(rf"(^|\.){r}\.", name):
            return r
    return "other"


def bits_per_weight(n_bytes: int, n_params: int) -> float:
    return 8.0 * n_bytes / n_params if n_params else 0.0


def layer_of(name: str):
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def analyse(path: Path) -> dict:
    from gguf import GGUFReader
    reader = GGUFReader(str(path))
    by_type, by_role = Counter(), defaultdict(Counter)
    params = nbytes = not_k_aligned = 0
    rows = []
    for t in reader.tensors:
        tname = t.tensor_type.name
        n, b = int(t.n_elements), int(t.n_bytes)
        params += n
        nbytes += b
        by_type[tname] += n
        if not t.name.endswith(".weight") or role(t.name) == "norm":
            continue  # biases and norms stay F32 in every file
        by_role[role(t.name)][tname] += 1
        rows.append((t.name, tname))
        if int(t.shape[0]) % QK_K:
            not_k_aligned += n
    other = {}
    for r, cnt in by_role.items():
        if len(cnt) > 1:
            major = cnt.most_common(1)[0][0]
            other[r] = sorted(layer_of(n) for n, tn in rows if role(n) == r and tn != major)
    return {
        "file": path.name,
        "size_mb": round(path.stat().st_size / 2**20, 1),
        "params_m": round(params / 1e6, 1),
        "bits_per_weight": round(bits_per_weight(nbytes, params), 2),
        "share_rows_not_multiple_of_256": round(not_k_aligned / params, 3),
        "share_by_type": {k: round(v / params, 3) for k, v in by_type.most_common()},
        "types_by_role": {r: dict(c) for r, c in sorted(by_role.items())},
        "layers_with_other_type": other,
    }


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "models" / "qwen25-05b-gguf"
    results = [analyse(p) for p in sorted(folder.glob("*.gguf"))]
    Path(__file__).with_name("results.json").write_text(json.dumps(results, indent=2))
    for r in results:
        print(f"{r['file']}: {r['size_mb']} MB, {r['params_m']}M params, {r['bits_per_weight']} bits/weight,"
              f" rows not a multiple of 256: {r['share_rows_not_multiple_of_256']}")
        print("  share by type:", r["share_by_type"])
        print("  types by role:", r["types_by_role"])
        print("  layers stored differently:", r["layers_with_other_type"])
