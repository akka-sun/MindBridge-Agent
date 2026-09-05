from __future__ import annotations

import argparse
import json
from pathlib import Path

from .deepseek_client import DeepSeekClient
from .io_utils import load_prompt, read_jsonl, write_jsonl
from .models import ClassifiedRecord, ClassificationReport, JointLabel, ProblemCandidate


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def build_classification_prompt(candidate: ProblemCandidate) -> str:
    history = "无" if not candidate.history else "\n".join(
        f"{message.role}: {message.content}" for message in candidate.history
    )
    return f"最近对话：\n{history}\n\n当前用户消息：\n{candidate.text}"


def classify_candidates(inputs: list[Path], output: Path, client: DeepSeekClient | None = None,
                        resume: bool = False) -> ClassificationReport:
    client = client or DeepSeekClient()
    version, system_prompt = load_prompt(PACKAGE_ROOT / "prompts/classify_joint.txt")
    schema = json.loads((PACKAGE_ROOT / "schemas/classification_response.schema.json").read_text(encoding="utf-8"))
    old_records = list(read_jsonl(output)) if resume and output.exists() else []
    existing = {x["candidate"]["id"] for x in old_records}
    records, failed, total = [], 0, 0
    for path in inputs:
        for raw in read_jsonl(path):
            total += 1
            candidate = ProblemCandidate.model_validate(raw)
            if candidate.id in existing:
                continue
            try:
                value = client.generate_json(system_prompt, build_classification_prompt(candidate), schema, temperature=.1)
                quality = {"confidence": value.pop("confidence", 1.0), "reason_codes": value.pop("reason_codes", [])}
                label = JointLabel.model_validate(value)
                records.append(ClassifiedRecord(
                    candidate=candidate, **label.model_dump(), confidence=quality.get("confidence", 1.0),
                    reason_codes=quality.get("reason_codes", []), classifier_model=client.model,
                    classifier_prompt_version=version,
                ))
            except Exception:
                failed += 1
    write_jsonl(output, [*old_records, *records])
    return ClassificationReport(total=total, classified=len(records), failed=failed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config")
    args = parser.parse_args()
    report = classify_candidates([Path(x) for x in args.input], Path(args.output), resume=args.resume)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
