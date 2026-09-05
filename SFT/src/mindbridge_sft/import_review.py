from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io_utils import read_jsonl, write_jsonl
from .models import ClassifiedRecord
from .review_export import import_review_csv


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--review",required=True);p.add_argument("--output",required=True);p.add_argument("--rejected-output");p.add_argument("--report");a=p.parse_args();records=[ClassifiedRecord.model_validate(x) for x in read_jsonl(a.data)];accepted,rejected=import_review_csv(records,a.review);write_jsonl(a.output,accepted)
    if a.rejected_output:write_jsonl(a.rejected_output,rejected)
    report={"accepted":len(accepted),"rejected":len(rejected)}
    if a.report:Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))


if __name__ == "__main__": main()

