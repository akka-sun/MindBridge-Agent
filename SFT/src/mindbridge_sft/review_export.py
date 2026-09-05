from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable

from .io_utils import read_jsonl, write_jsonl
from .models import ClassifiedRecord, ReviewedRecord


REVIEW_STATES = {"ACCEPT", "ACCEPT_AFTER_EDIT", "REJECT_PRIVACY", "REJECT_DANGEROUS_DETAIL",
                 "REJECT_UNNATURAL", "REJECT_AMBIGUOUS", "REJECT_LABEL_UNRESOLVABLE"}
FIELDS = ["id", "text", "history_json", "intent", "emotion", "risk", "confidence", "reason_codes",
          "review_status", "reviewer_ids", "reviewed_at", "review_note"]


def select_for_review(records: list[ClassifiedRecord], seed: int = 42) -> list[ClassifiedRecord]:
    rng = random.Random(seed)
    selected = []
    for record in records:
        candidate = record.candidate
        rate = 1.0 if record.intent.value == "RISK" or candidate.is_redteam else (.3 if candidate.boundary_type else .1)
        if rng.random() < rate:
            selected.append(record)
    return selected


def export_review_csv(records: Iterable[ClassifiedRecord], output: str | Path) -> None:
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader()
        for record in records:
            writer.writerow({
                "id": record.candidate.id, "text": record.candidate.text,
                "history_json": json.dumps([x.model_dump() for x in record.candidate.history], ensure_ascii=False),
                "intent": record.intent.value, "emotion": record.emotion.value, "risk": record.risk.value,
                "confidence": record.confidence, "reason_codes": "|".join(record.reason_codes),
                "review_status": "", "reviewer_ids": "", "reviewed_at": "", "review_note": "",
            })


def import_review_csv(records: Iterable[ClassifiedRecord], review_path: str | Path) -> tuple[list[ReviewedRecord], list[dict]]:
    source = {x.candidate.id: x for x in records}; accepted, rejected, reviewed_ids = [], [], set()
    with Path(review_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            status = row["review_status"].strip().upper()
            if status not in REVIEW_STATES: raise ValueError(f"invalid review status for {row['id']}: {status}")
            base = source[row["id"]]
            reviewed_ids.add(row["id"])
            data = base.model_dump()
            for field in ("intent", "emotion", "risk"):
                if row.get(field): data[field] = row[field].strip().upper()
            reviewed = ReviewedRecord.model_validate({**data, "review_status":status,
                "reviewer_ids":[x for x in row.get("reviewer_ids", "").split("|") if x],
                "reviewed_at":row.get("reviewed_at") or None, "review_note":row.get("review_note", "")})
            if reviewed.intent.value == "RISK" and not reviewed.reviewer_ids:
                raise ValueError(f"RISK record {row['id']} requires reviewer_ids")
            if status in {"ACCEPT", "ACCEPT_AFTER_EDIT"}: accepted.append(reviewed)
            else: rejected.append({"id":row["id"], "review_status":status, "review_note":row.get("review_note", "")})
    for record_id, base in source.items():
        if record_id in reviewed_ids:
            continue
        mandatory = base.intent.value == "RISK" or base.candidate.is_redteam or bool(base.candidate.boundary_type) or base.confidence < .6
        if mandatory:
            raise ValueError(f"mandatory review record is missing from review CSV: {record_id}")
        accepted.append(ReviewedRecord.model_validate({**base.model_dump(), "review_status":"AUTO_ACCEPTED"}))
    return accepted, rejected


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export"); export.add_argument("--input", required=True); export.add_argument("--output", required=True); export.add_argument("--seed", type=int, default=42)
    imp = sub.add_parser("import"); imp.add_argument("--data", required=True); imp.add_argument("--review", required=True); imp.add_argument("--output", required=True); imp.add_argument("--rejected-output")
    args = parser.parse_args(); records = [ClassifiedRecord.model_validate(x) for x in read_jsonl(args.input if args.command == "export" else args.data)]
    if args.command == "export": export_review_csv(select_for_review(records, args.seed), args.output)
    else:
        accepted, rejected = import_review_csv(records, args.review); write_jsonl(args.output, accepted)
        if args.rejected_output: write_jsonl(args.rejected_output, rejected)


if __name__ == "__main__": main()
