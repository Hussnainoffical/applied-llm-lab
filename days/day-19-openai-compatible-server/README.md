# Day 19: an OpenAI-compatible endpoint and what "JSON mode" has to fix

New-tech track, step 2. `server.py` is a stdlib-only `/v1/chat/completions` endpoint (OpenAI request and
response shape, `usage` token counts) in front of the official Qwen2.5-0.5B-Instruct **Q8_0 GGUF**, loaded with
`transformers` (`gguf_file=`, de-quantized to fp32 on CPU). `response_format={"type": "json_object"}` is enforced
after generation. `bench.py` sends 12 extraction requests over plain HTTP ("return JSON with keys name, city,
date" for 12 synthetic meetup sentences) with and without JSON mode, greedy decoding, `max_tokens` 80.

## Result (measured 17 Nov 2026, Windows, CPU, torch 2.15 CPU, transformers 5.23)

| Run | JSON mode does | Valid JSON | Exact keys | Retries | Mean latency |
|---|---|---|---|---|---|
| plain (both runs) | nothing | 0 / 12 | 0 / 12 | 0 | 18.7 s (run 2), 21.3 s (run 1) |
| run 1 | validate, on failure retry once with the parse error fed back | 0 / 12 | 0 / 12 | 12 | 41.3 s |
| run 2 | strip a Markdown code fence, then validate (retry kept as fallback) | 12 / 12 | 12 / 12 | 0 | 13.4 s |

Raw outputs: `results-run1-no-fence-strip.json` and `results.json`.

## What I take from it

- The model got the content right and still failed every plain request: all 12 answers were correct JSON wrapped
  in a ```` ```json ```` fence (`"fenced": 12`). A strict `json.loads` treats a formatting habit as a wrong answer.
- Feeding the parse error back did not help at all. Every one of the 12 retries came back fenced again, so the
  retry only doubled latency (41 s instead of 21 s) and token cost for zero gain.
- A two-line server-side normalisation (strip one surrounding fence, then validate) took valid JSON from 0 to
  12 of 12 with no retries. Fix the deterministic part in code before spending a second model call on it.
- Exact keys here means the three requested keys and nothing else; I did not score the values themselves.
- Speed is not representative: `transformers` de-quantizes the GGUF to fp32, so this ran at about 2 to 3 completion
  tokens per second on a 4-thread CPU. Measuring real GGUF speed needs llama.cpp, which is still blocked on
  Windows (no llama-cpp-python wheel on PyPI, no compiler; see Day 18).

## Run

```
pip install torch transformers accelerate gguf pytest
pytest -q                       # 6 tests, no model needed (fake generator)
python bench.py qwen2.5-0.5b-instruct-q8_0.gguf   # GGUF in ~/models/qwen25-05b-gguf
```
