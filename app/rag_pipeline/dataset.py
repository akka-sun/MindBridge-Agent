from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

from .models import EvaluationCase


def quota_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(row.get("question_type", "") for row in rows))


def validate_evidence_quote(quote: str, section_text: str) -> bool:
    return bool(quote.strip()) and quote.strip() in section_text


class DatasetValidator:
    def __init__(self, sections: dict[str, str]):
        self.sections = sections

    def validate(self, rows: list[dict], exact_count: int = 300) -> list[EvaluationCase]:
        if len(rows) != exact_count:
            raise ValueError(f"expected exactly {exact_count} cases, got {len(rows)}")
        cases = [EvaluationCase.model_validate(row) for row in rows]
        seen = set()
        for case in cases:
            if case.id in seen:
                raise ValueError(f"duplicate case id: {case.id}")
            seen.add(case.id)
            for evidence in case.gold_evidence:
                if not validate_evidence_quote(evidence.quote, self.sections.get(evidence.section_id, "")):
                    raise ValueError(f"evidence quote not found: {case.id}/{evidence.section_id}")
        return cases


class DatasetGenerator:
    def __init__(self, client):
        self.client = client

    def generate_batch(self, prompt_id: str, evidence: str, quota: dict[str, int]) -> list[dict]:
        response = self.client.generate_json(prompt_id, evidence, {})
        return response.get("questions", [])

