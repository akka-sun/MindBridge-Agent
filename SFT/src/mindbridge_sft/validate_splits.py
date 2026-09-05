from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io_utils import file_sha256, read_jsonl


def validate_directory(directory: Path) -> dict:
    files=[directory/"train.jsonl",directory/"valid.jsonl",directory/"test.jsonl",directory/"redteam_test.jsonl"];groups={};hashes={}
    for path in files:
        if not path.exists(): raise FileNotFoundError(path)
        hashes[path.name]=file_sha256(path);groups[path.name]={((x.get("candidate") or x).get("split_group") or (x.get("candidate") or x)["id"]) for x in read_jsonl(path)}
    overlaps={}
    for i,left in enumerate(files):
        for right in files[i+1:]:
            shared=groups[left.name]&groups[right.name]
            if shared:overlaps[f"{left.name}:{right.name}"]=sorted(shared)
    return {"valid":not overlaps,"overlaps":overlaps,"hashes":hashes}


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--data-dir",required=True);p.add_argument("--report",required=True);a=p.parse_args();report=validate_directory(Path(a.data_dir));Path(a.report).parent.mkdir(parents=True,exist_ok=True);Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2));raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__": main()
