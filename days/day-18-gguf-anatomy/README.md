# Day 18: what is inside a Q4_0, Q4_K_M and Q8_0 GGUF file

New-tech track, step 1. Before timing llama.cpp on a CPU I wanted to know what the quantization
names actually mean on disk. `gguf_anatomy.py` reads the tensor table of three official
[Qwen2.5-0.5B-Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) files with the
`gguf` package from PyPI (no inference, no llama.cpp build) and counts bits per weight, the type of
every weight matrix by role, and which layers a mixed K-quant stores differently.

## Result (measured 15 Nov 2026, Windows, `gguf` from PyPI, Python 3.13)

| File | Size | Bits per weight | Share of weights by type |
|---|---|---|---|
| q4_0 | 408.9 MB | 5.37 | Q4_0 78.4%, Q8_0 21.6% (the output matrix) |
| q4_k_m | 468.6 MB | 6.16 | Q5_0 61.6%, Q8_0 21.8%, Q6_K 8.3%, Q4_K 8.3% |
| q8_0 | 644.4 MB | 8.50 | Q8_0 100% |

630.2M parameters in every file. Norms and the q/k/v biases stay F32 everywhere.

## What I take from it

- On this model "Q4_K_M" is mostly not 4-bit and not K-quant: only 8.3% of the weights are Q4_K.
  83.4% of the weights sit in matrices whose rows are 896 long, and K-quants pack rows in super-blocks
  of 256 (896 = 3.5 x 256). Those matrices fall back to Q5_0 (and Q8_0 where Q6_K was wanted). Only
  `ffn_down` (rows of 4864 = 19 x 256) gets real K-quants.
- So the Q4_K_M file is 15% larger than Q4_0 (6.16 vs 5.37 bits per weight). A size or speed comparison
  of quant names across models with different hidden sizes compares different things.
- "M" is a per-layer mix: 12 of 24 layers keep `ffn_down` at Q6_K and `attn_v` at Q8_0, the other 12
  (layers 2, 4, 5, 11, 12, 14, 15, 17, 18, 20, 22, 23) use Q4_K and Q5_0.
- The 151,936 x 896 output matrix alone is about 21.6% of all weights and stays Q8_0 even in the Q4_0
  file, which is why "4-bit" Q4_0 lands at 5.37 bits per weight on a small model with a big vocabulary.

## Run

```
python -m venv %USERPROFILE%\.venvs\gguf
%USERPROFILE%\.venvs\gguf\Scripts\pip install gguf numpy pytest huggingface_hub
# download the three .gguf files from Qwen/Qwen2.5-0.5B-Instruct-GGUF into ~/models/qwen25-05b-gguf
python gguf_anatomy.py          # writes results.json
python -m pytest -q             # 3 passed (the real-file test skips if the files are missing)
```

Next: tokens per second and time to first token for Q4_0 vs Q4_K_M vs Q8_0 on this CPU, once a
llama.cpp runtime is available on the PC (llama-cpp-python has no Windows wheel on PyPI and there is
no C++ compiler on the machine).
