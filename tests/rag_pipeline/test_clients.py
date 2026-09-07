import json
from pathlib import Path

import httpx

from app.rag_pipeline.clients import DeepSeekClient, JsonFileCache, SiliconFlowClient


def test_deepseek_json_request_and_cache(tmp_path: Path):
    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek-v4-flash"
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":true}'}}]})

    transport = httpx.MockTransport(handler)
    client = DeepSeekClient("key", "https://api.deepseek.com", "deepseek-v4-flash", transport=transport, cache=JsonFileCache(tmp_path))
    assert client.generate_json("p", "s", {"x": 1}) == {"ok": True}
    assert client.generate_json("p", "s", {"x": 1}) == {"ok": True}
    assert len(calls) == 1


def test_siliconflow_embed_parses_ordered_vectors():
    def handler(request: httpx.Request):
        return httpx.Response(200, json={"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]})

    client = SiliconFlowClient("key", "https://api.siliconflow.cn/v1", "embed", "rerank", transport=httpx.MockTransport(handler))
    assert client.embed(["a", "b"]) == [[1.0], [2.0]]
