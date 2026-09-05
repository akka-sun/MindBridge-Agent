from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Literal

import yaml

from .deepseek_client import DeepSeekClient
from .io_utils import append_jsonl, load_prompt, read_jsonl
from .models import GenerationReport, ProblemCandidate


ROOT = Path(__file__).resolve().parents[2]


def generate_candidates(kind: Literal["ordinary", "boundary"], config: dict[str, Any], output: Path,
                        client: DeepSeekClient | None = None, count: int | None = None,
                        resume: bool = False, dry_run: bool = False) -> GenerationReport:
    client = client or DeepSeekClient()
    section = config[kind]
    target = count or int(section["target_count"])
    batch_size = int(section.get("batch_size", 20))
    prompt_name = "ordinary_problem.txt" if kind == "ordinary" else "boundary_problem.txt"
    schema_name = "ordinary_problem.schema.json" if kind == "ordinary" else "boundary_problem.schema.json"
    version, system_prompt = load_prompt(ROOT / "prompts" / prompt_name)
    schema = json.loads((ROOT / "schemas" / schema_name).read_text(encoding="utf-8"))
    existing = list(read_jsonl(output)) if resume and output.exists() else []
    ids = {x["id"] for x in existing}
    completed = {x.get("generation_batch_id") for x in existing}
    generated, failures, skipped = 0, 0, 0
    failed_path = output.with_name("failed_batches.jsonl")
    scenarios = section.get("scenes") or section.get("boundary_types") or ["综合"]
    for offset in range(0, target, batch_size):
        amount = min(batch_size, target - offset)
        scene = scenarios[(offset // batch_size) % len(scenarios)]
        batch_id = f"{kind}-{offset // batch_size:05d}"
        if batch_id in completed:
            skipped += 1
            continue
        user_prompt = f"本批次主题：{scene}\n目标数量：{amount}\n语言风格：自然口语\n是否允许历史上下文：是"
        if dry_run:
            print(user_prompt)
            continue
        try:
            values = client.generate_json(system_prompt, user_prompt, schema,
                                          temperature=float(section.get("temperature", .8)),
                                          top_p=float(section.get("top_p", .9)))
            if not isinstance(values, list):
                raise ValueError("question generator must return a JSON array")
            prepared = []
            for value in values:
                candidate = ProblemCandidate.model_validate(value)
                if candidate.id in ids:
                    raise ValueError(f"duplicate id: {candidate.id}")
                ids.add(candidate.id)
                candidate.generation_batch_id = batch_id
                candidate.prompt_version = version
                candidate.generator_model = client.model
                candidate.generation_parameters = {"temperature": section.get("temperature", .8), "top_p": section.get("top_p", .9)}
                candidate.split_group = batch_id
                prepared.append(candidate)
            append_jsonl(output, prepared)
            generated += len(prepared)
        except Exception as exc:
            failures += 1
            append_jsonl(failed_path, [{"batch_id":batch_id, "kind":kind, "scene":scene, "error":repr(exc)}])
    return GenerationReport(requested=target, generated=generated, skipped_batches=skipped, failed_batches=failures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", "--question-type", choices=["ordinary", "boundary"], required=True)
    parser.add_argument("--config", default="configs/data_generation.yaml")
    parser.add_argument("--output", default="data/raw/candidates.jsonl")
    parser.add_argument("--count", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    report = generate_candidates(args.kind, config, Path(args.output), count=args.count, resume=args.resume, dry_run=args.dry_run)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__": main()

