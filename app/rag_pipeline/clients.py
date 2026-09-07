from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Callable

import httpx


class ApiError(RuntimeError):
    pass


class JsonFileCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, payload: object) -> str:
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def get(self, key: str):
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value: object) -> None:
        (self.root / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class UsageLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


class _BaseClient:
    def __init__(self, api_key: str, transport=None, timeout: float = 120, max_retries: int = 5, sleep: Callable[[float], None] = time.sleep):
        self.api_key = api_key
        self.client = httpx.Client(transport=transport, timeout=timeout)
        self.max_retries = max(0, max_retries)
        self.sleep = sleep

    def _request(self, method: str, url: str, payload: dict) -> dict:
        if not self.api_key:
            raise ApiError("API key is not configured")
        for attempt in range(self.max_retries + 1):
            response = self.client.request(method, url, headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, json=payload)
            if response.status_code < 400:
                return response.json()
            if response.status_code not in {408, 425, 429, 500, 502, 503, 504} or attempt >= self.max_retries:
                raise ApiError(f"API request failed with status {response.status_code}: {response.text[:300]}")
            self.sleep(min(30.0, (2 ** attempt) + random.random()))
        raise ApiError("API request failed after retries")


class DeepSeekClient(_BaseClient):
    def __init__(self, api_key: str, base_url: str, model: str, *, transport=None, cache: JsonFileCache | None = None, timeout: float = 120, max_retries: int = 5, sleep=time.sleep):
        super().__init__(api_key, transport, timeout, max_retries, sleep)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cache = cache

    def generate_json(self, prompt_id: str, prompt: str, schema: dict, parameters: dict | None = None) -> dict:
        parameters = parameters or {}
        structured_prompt = prompt if "json" in prompt.lower() else prompt + "\n请严格输出有效JSON。"
        payload = {"model": self.model, "messages": [{"role": "user", "content": structured_prompt}], "response_format": {"type": "json_object"}, **parameters}
        cache_key = self.cache.key({"provider": "deepseek", "prompt_id": prompt_id, "payload": payload, "schema": schema}) if self.cache else None
        if cache_key and (value := self.cache.get(cache_key)) is not None:
            return value
        response = self._request("POST", f"{self.base_url}/chat/completions", payload)
        try:
            content = response["choices"][0]["message"]["content"]
            value = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ApiError("DeepSeek returned malformed JSON response") from exc
        if cache_key:
            self.cache.put(cache_key, value)
        return value


class SiliconFlowClient(_BaseClient):
    def __init__(self, api_key: str, base_url: str, embedding_model: str, rerank_model: str, *, transport=None, cache: JsonFileCache | None = None, timeout: float = 120, max_retries: int = 5, sleep=time.sleep):
        super().__init__(api_key, transport, timeout, max_retries, sleep)
        self.base_url = base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.cache = cache

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.embedding_model, "input": texts, "encoding_format": "float"}
        key = self.cache.key({"kind": "embed", "payload": payload}) if self.cache else None
        if key and (value := self.cache.get(key)) is not None:
            return value
        response = self._request("POST", f"{self.base_url}/embeddings", payload)
        try:
            rows = sorted(response["data"], key=lambda item: item["index"])
            vectors = [[float(x) for x in row["embedding"]] for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError("SiliconFlow embedding response is malformed") from exc
        if len(vectors) != len(texts):
            raise ApiError("SiliconFlow embedding count does not match input")
        if key:
            self.cache.put(key, vectors)
        return vectors

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[dict]:
        payload = {"model": self.rerank_model, "query": query, "documents": documents, "return_documents": True, "top_n": top_n}
        response = self._request("POST", f"{self.base_url}/rerank", payload)
        rows = response.get("results")
        if not isinstance(rows, list):
            raise ApiError("SiliconFlow rerank response is malformed")
        return rows
