import json
import threading
import urllib.request

from server import complete, serve, strip_fences


def fake(outputs):
    calls = []

    def generate(messages, max_tokens):
        calls.append(messages)
        return outputs[len(calls) - 1], 10, 5
    return generate, calls


def test_plain_request_has_openai_shape():
    gen, calls = fake(["hello"])
    r = complete({"messages": [{"role": "user", "content": "hi"}]}, gen)
    assert r["object"] == "chat.completion"
    assert r["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert r["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert len(calls) == 1


def test_json_mode_retries_once_with_error():
    gen, calls = fake(['{"name": "Anna",', '{"name": "Anna"}'])
    r = complete({"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_object"}}, gen)
    assert r["x_attempts"] == 2
    assert json.loads(r["choices"][0]["message"]["content"]) == {"name": "Anna"}
    assert "not valid JSON" in calls[1][-1]["content"]
    assert r["usage"]["total_tokens"] == 30


def test_json_mode_skips_retry_when_valid():
    gen, calls = fake(['{"a": 1}'])
    r = complete({"messages": [], "response_format": {"type": "json_object"}}, gen)
    assert r["x_attempts"] == 1 and len(calls) == 1


def test_http_roundtrip():
    gen, _ = fake(["ok"])
    server = serve(gen, port=8091)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    req = urllib.request.Request("http://127.0.0.1:8091/v1/chat/completions",
                                 json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                                 {"Content-Type": "application/json"})
    body = json.loads(urllib.request.urlopen(req, timeout=10).read())
    server.shutdown()
    assert body["choices"][0]["message"]["content"] == "ok"


def test_fenced_json_is_accepted_without_retry():
    gen, calls = fake(['```json\n{"name": "Anna"}\n```'])
    r = complete({"messages": [], "response_format": {"type": "json_object"}}, gen)
    assert r["x_attempts"] == 1 and json.loads(r["choices"][0]["message"]["content"]) == {"name": "Anna"}


def test_strip_fences_leaves_plain_text():
    assert strip_fences('{"a": 1}') == '{"a": 1}'
    assert strip_fences('```\n[1, 2]\n```') == '[1, 2]'
