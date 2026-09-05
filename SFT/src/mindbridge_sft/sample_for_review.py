from __future__ import annotations

import argparse

from .io_utils import read_jsonl
from .models import ClassifiedRecord
from .review_export import export_review_csv, select_for_review


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",required=True);p.add_argument("--seed",type=int,default=42);p.add_argument("--ordinary-count",type=int);p.add_argument("--boundary-count",type=int);a=p.parse_args();records=[ClassifiedRecord.model_validate(x) for x in read_jsonl(a.input)];export_review_csv(select_for_review(records,a.seed),a.output)


if __name__ == "__main__": main()

