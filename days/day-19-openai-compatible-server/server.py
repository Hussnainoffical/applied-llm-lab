"""Minimal OpenAI-compatible /v1/chat/completions endpoint over a local GGUF model (transformers, CPU).

response_format={"type": "json_object"} is enforced after generation: invalid JSON gets one retry
with the parse error fed back. The generator is injectable so tests run without a model.
"""
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def load_generator(model_dir, gguf_file):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, gguf_file=gguf_file)
    model = AutoModelForCausalLM.from_pretrained(model_dir, gguf_file=gguf_file)

    def generate(messages, max_tokens):
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=False)
        new = out[0, ids["input_ids"].shape[1]:]
        return tok.decode(new, skip_special_tokens=True), ids["input_ids"].shape[1], len(new)
    return generate


FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def strip_fences(text):
    m = FENCE.match(text)
    return m.group(1) if m else text


def complete(body, generate):
    messages = list(body["messages"])
    want_json = (body.get("response_format") or {}).get("type") == "json_object"
    text, p_tok, c_tok = generate(messages, body.get("max_tokens", 128))
    attempts = 1
    if want_json:
        text = strip_fences(text)
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"That was not valid JSON ({e.msg}). Reply with only the JSON object."}]
            text, p2, c2 = generate(messages, body.get("max_tokens", 128))
            text = strip_fences(text)
            p_tok, c_tok, attempts = p_tok + p2, c_tok + c2, 2
    return {"id": f"chatcmpl-{int(time.time() * 1000)}", "object": "chat.completion", "created": int(time.time()),
            "model": body.get("model", "local"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok},
            "x_attempts": attempts}


def serve(generate, port=8019):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps(complete(body, generate)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
