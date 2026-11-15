from pathlib import Path

import pytest

from gguf_anatomy import analyse, bits_per_weight, layer_of, role

MODELS = Path.home() / "models" / "qwen25-05b-gguf"


def test_role_mapping():
    assert role("token_embd.weight") == "token_embd"
    assert role("blk.3.attn_v.weight") == "attn_v"
    assert role("blk.3.attn_output.weight") == "attn_output"
    assert role("blk.0.ffn_down.weight") == "ffn_down"
    assert role("blk.0.attn_norm.weight") == "norm"
    assert role("output.weight") == "output"
    assert layer_of("blk.17.ffn_up.weight") == 17


def test_bits_per_weight():
    # Q8_0 block: 32 int8 weights + one fp16 scale = 34 bytes -> 8.5 bits per weight
    assert bits_per_weight(34, 32) == pytest.approx(8.5)
    assert bits_per_weight(0, 0) == 0.0


@pytest.mark.skipif(not (MODELS / "qwen2.5-0.5b-instruct-q8_0.gguf").exists(), reason="model files not downloaded")
def test_real_files_order():
    q8 = analyse(MODELS / "qwen2.5-0.5b-instruct-q8_0.gguf")
    q4km = analyse(MODELS / "qwen2.5-0.5b-instruct-q4_k_m.gguf")
    assert q8["params_m"] == q4km["params_m"]
    assert q4km["bits_per_weight"] < q8["bits_per_weight"]
