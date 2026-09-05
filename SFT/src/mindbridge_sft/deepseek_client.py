from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from jsonschema import validate

from .io_utils import append_jsonl


class DeepSeekError(RuntimeError):
    pass


class DeepSeekClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, timeout: float = 60, max_retries: int = 3,
                 audit_path: str | Path = "data/raw/api_audit.jsonl",
                 transport: httpx.BaseTransport | None = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.max_retries = max_retries
        self.audit_path = Path(audit_path)
        self.http = httpx.Client(timeout=timeout, transport=transport)

    def generate_json(self, system_prompt: str, user_prompt: str, schema: dict[str, Any],
                      temperature: float = .2, top_p: float = .9,
                      max_tokens: int = 2048) -> dict[str, Any] | list[dict[str, Any]]:
        if not self.api_key:
            raise DeepSeekError("DEEPSEEK_API_KEY is not configured")
        request_id = str(uuid.uuid4())
        payload = {
            "model": self.model,
            "messages": [{"role":"system","content":system_prompt}, {"role":"user","content":user_prompt}],
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
        }
        if schema.get("type") == "object":
            payload["response_format"] = {"type":"json_object"}
        prompt_hash = hashlib.sha256((system_prompt + "\0" + user_prompt).encode()).hexdigest()
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            raw = ""
            try:
                response = self.http.post(f"{self.base_url}/chat/completions", json=payload,
                                          headers={"Authorization": f"Bearer {self.api_key}", "X-Request-ID": request_id})
                if response.status_code in (401, 403):
                    response.raise_for_status()
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                body = response.json()
                raw = body["choices"][0]["message"]["content"]
                parsed = self._parse_json(raw)
                validate(parsed, schema)
                self._audit(request_id, prompt_hash, payload, raw, attempt, None)
                return parsed
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError,
                    json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                self._audit(request_id, prompt_hash, payload, raw, attempt, repr(exc))
                non_retryable = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (400, 401, 403, 404, 422)
                if non_retryable or attempt == self.max_retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))
        raise DeepSeekError(f"DeepSeek request failed after {attempt} attempt(s): {last_error}") from last_error

    @staticmethod
    def _parse_json(raw: str) -> Any:
        value = raw.strip()
        if value.startswith("```"):
            lines = value.splitlines()
            value = "\n".join(lines[1:-1])
        return json.loads(value)

    def _audit(self, request_id: str, prompt_hash: str, payload: dict[str, Any], raw: str,
               attempt: int, error: str | None) -> None:
        append_jsonl(self.audit_path, [{
            "request_id": request_id, "model": self.model, "prompt_hash": prompt_hash,
            "parameters": {k: payload[k] for k in ("temperature", "top_p", "max_tokens")},
            "attempt": attempt, "raw_response": raw, "error": error,
        }])
