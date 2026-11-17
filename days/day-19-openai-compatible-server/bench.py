"""Start the server on a local GGUF, send 12 extraction requests over plain HTTP with and without
response_format=json_object, and measure valid JSON, exact keys, retries and latency."""
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

from server import load_generator, serve

SENTENCES = [
    "The PyData meetup in Berlin takes place on 2026-12-03, hosted by Anna.",
    "On 2027-01-15 Jonas runs a workshop on vector search in Munich.",
    "Hamburg will see a talk on agent testing by Lea on 2026-11-28.",
    "Cologne, 2027-02-02: Tim presents his LLM evaluation results.",
    "Mia organises an MLOps evening in Frankfurt on 2026-12-10.",
    "A Haystack hands-on session led by Paul is set for 2027-03-04 in Leipzig.",
    "Stuttgart hosts Sara's talk about quantized models on 2026-12-17.",
    "On 2027-01-20 Ben gives a talk on speech recognition in Dresden.",
    "The computer vision meetup in Bonn on 2027-02-11 is run by Lena.",
    "Max demos n8n workflows in Hannover on 2026-12-01.",
    "In Nuremberg, Eva speaks about retrieval evaluation on 2027-03-18.",
    "Karl presents a GGUF deep dive in Karlsruhe on 2027-01-08.",
]
PROMPT = "Extract the person's name, the city and the date from this sentence. Return JSON with keys name, city, date.\n\n"


def call(port, sentence, json_mode):
    body = {"model": "qwen2.5-0.5b", "max_tokens": 80, "messages": [{"role": "user", "content": PROMPT + sentence}]}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.perf_counter()
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return resp, time.perf_counter() - t


def score(text):
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False, False
    return True, isinstance(obj, dict) and set(obj) == {"name", "city", "date"}


if __name__ == "__main__":
    gguf = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-0.5b-instruct-q8_0.gguf"
    server = serve(load_generator(str(Path.home() / "models" / "qwen25-05b-gguf"), gguf))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    results = {"gguf": gguf}
    for mode in (False, True):
        rows = [call(8019, s, mode) for s in SENTENCES]
        ok = [score(r["choices"][0]["message"]["content"]) for r, _ in rows]
        tok = sum(r["usage"]["completion_tokens"] for r, _ in rows)
        secs = sum(t for _, t in rows)
        results["json_object" if mode else "plain"] = {
            "valid_json": sum(v for v, _ in ok), "exact_keys": sum(k for _, k in ok), "n": len(rows),
            "retries": sum(r["x_attempts"] - 1 for r, _ in rows),
            "fenced": sum(r["choices"][0]["message"]["content"].lstrip().startswith("```") for r, _ in rows), "mean_latency_s": round(secs / len(rows), 2),
            "completion_tok_per_s": round(tok / secs, 1),
            "example": rows[0][0]["choices"][0]["message"]["content"]}
    server.shutdown()
    print(json.dumps(results, indent=2))
    Path("results.json").write_text(json.dumps(results, indent=2))
