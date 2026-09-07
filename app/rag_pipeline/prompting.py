from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

import yaml
from jsonschema import Draft202012Validator


class PromptAssetError(ValueError):
    pass


@dataclass(frozen=True)
class PromptAsset:
    prompt_id: str
    version: str
    model_family: str
    schema_path: Path
    temperature: float
    max_output_tokens: int
    body: str

    @classmethod
    def load(cls, prompt_id: str, root: Path | None = None) -> "PromptAsset":
        root = root or Path(__file__).resolve().parent
        path = root / "prompts" / f"{prompt_id}.md"
        if not path.exists():
            raise PromptAssetError(f"prompt not found: {prompt_id}")
        raw = path.read_text(encoding="utf-8")
        if not raw.startswith("---"):
            raise PromptAssetError(f"prompt metadata missing: {prompt_id}")
        _, front, body = raw.split("---", 2)
        metadata = yaml.safe_load(front) or {}
        required = {"prompt_id", "version", "model_family", "schema", "temperature", "max_output_tokens"}
        missing = required - set(metadata)
        if missing:
            raise PromptAssetError(f"prompt metadata missing fields: {sorted(missing)}")
        if metadata["prompt_id"] != prompt_id:
            raise PromptAssetError("prompt_id does not match filename")
        if not re.search(r"普通|ordinary", body, re.I) or not re.search(r"边界|boundary", body, re.I):
            raise PromptAssetError(f"prompt must include ordinary and boundary examples: {prompt_id}")
        if not re.search(r"拒绝|失败|failure|reject", body, re.I):
            raise PromptAssetError(f"prompt must include failure example: {prompt_id}")
        schema_path = (path.parent / metadata["schema"]).resolve()
        if not schema_path.exists():
            raise PromptAssetError(f"schema not found: {schema_path}")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        return cls(prompt_id, str(metadata["version"]), str(metadata["model_family"]), schema_path,
                   float(metadata["temperature"]), int(metadata["max_output_tokens"]), body.strip())

    def render(self, values: dict[str, object]) -> str:
        safe = {key: f"<data name=\"{key}\">{value}</data>" for key, value in values.items()}
        return Template(self.body).safe_substitute(safe)

    def cache_fingerprint(self, model: str, parameters: dict[str, object] | None = None) -> str:
        payload = {"body": self.body, "version": self.version, "schema": self.schema_path.read_text(encoding="utf-8"),
                   "model": model, "parameters": parameters or {}}
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

