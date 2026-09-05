from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .io_utils import file_sha256, read_jsonl, write_jsonl
from .models import ReviewedRecord


def split_records(records: list[ReviewedRecord], sizes: dict[str, int], seed: int = 42) -> dict[str, list[ReviewedRecord]]:
    required = sum(sizes.values())
    if len(records) < required: raise ValueError(f"insufficient reviewed data: need {required}, have {len(records)}, gap {required-len(records)}")
    groups: dict[str, list[ReviewedRecord]] = {}
    for record in records:
        group = record.candidate.split_group or record.candidate.generation_batch_id or record.candidate.id
        groups.setdefault(group, []).append(record)
    rng = random.Random(seed)
    ordered = list(groups.items()); rng.shuffle(ordered)
    ordered.sort(key=lambda item: len(item[1]), reverse=True)
    result = {name: [] for name in sizes}
    for _, values in ordered:
        eligible = [name for name, limit in sizes.items() if len(result[name]) + len(values) <= limit]
        if not eligible: continue
        def score(name):
            fill = len(result[name]) / max(sizes[name], 1)
            same_intent = sum(x.intent == values[0].intent for x in result[name])
            return fill + same_intent / max(sizes[name], 1)
        result[min(eligible, key=score)].extend(values)
    gaps = {name: sizes[name] - len(values) for name, values in result.items() if len(values) != sizes[name]}
    if gaps: raise ValueError(f"cannot satisfy exact split sizes without splitting groups: {gaps}")
    return result


def export_splits(splits: dict[str, list[ReviewedRecord]], output_dir: Path, seed: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True); hashes = {}; manifest = {"seed":seed, "splits":{}}
    names = {"redteam":"redteam_test.jsonl"}
    for name, values in splits.items():
        path = output_dir / names.get(name, f"{name}.jsonl"); write_jsonl(path, values); hashes[path.name] = file_sha256(path)
        manifest["splits"][name] = {"count":len(values), "intent":dict(Counter(x.intent.value for x in values))}
    (output_dir / "split_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--train-size",type=int,default=4200);p.add_argument("--valid-size",type=int,default=900);p.add_argument("--test-size",type=int,default=900);p.add_argument("--redteam-size",type=int,default=600);p.add_argument("--seed",type=int,default=42);p.add_argument("--report");p.add_argument("--stratify-by",nargs="*");p.add_argument("--group-by")
    a=p.parse_args(); records=[ReviewedRecord.model_validate(x) for x in read_jsonl(a.input)]; sizes={"train":a.train_size,"valid":a.valid_size,"test":a.test_size,"redteam":a.redteam_size}; result=split_records(records,sizes,a.seed); manifest=export_splits(result,Path(a.output_dir),a.seed); print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()

