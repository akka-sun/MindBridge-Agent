from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from .io_utils import read_jsonl, write_jsonl
from .models import ClassifiedRecord, QualityReport, ReviewIssue
from .quality_rules import has_dangerous_detail, has_meta_language, label_conflicts, normalize_text, privacy_codes
from jsonschema import validate as validate_schema


def validate_records(records: Iterable[ClassifiedRecord], confidence_threshold: float = .6):
    accepted, review, rejected, seen = [], [], [], set()
    duplicate_count = 0
    source = list(records)
    for record in source:
        rid, text = record.candidate.id, record.candidate.text
        reject_codes = privacy_codes(text)
        if has_dangerous_detail(text): reject_codes.append("DANGEROUS_DETAIL")
        if has_meta_language(text): reject_codes.append("META_LANGUAGE")
        normalized = normalize_text(text)
        if normalized in seen:
            reject_codes.append("DUPLICATE")
            duplicate_count += 1
        seen.add(normalized)
        if reject_codes:
            rejected.extend(ReviewIssue(record_id=rid, code=code, severity="REJECT", message=code) for code in reject_codes)
            continue
        review_codes = label_conflicts(record)
        if record.confidence < confidence_threshold: review_codes.append("LOW_CONFIDENCE")
        if record.candidate.boundary_type: review_codes.append("BOUNDARY_SAMPLE")
        if review_codes:
            issues = [ReviewIssue(record_id=rid, code=code, severity="REVIEW", message=code) for code in review_codes]
            review.extend(issues)
        else:
            accepted.append(record)
    distributions = {
        name: dict(Counter(getattr(record, name).value for record in source))
        for name in ("intent", "emotion", "risk")
    }
    report = QualityReport(total=len(source), accepted=len(accepted), needs_review=len({x.record_id for x in review}),
                           rejected_count=len({x.record_id for x in rejected}), duplicate_count=duplicate_count,
                           rejected=rejected, review_issues=review, distributions=distributions)
    return accepted, review, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output")
    parser.add_argument("--rejected-output")
    parser.add_argument("--report", required=True)
    parser.add_argument("--schema")
    parser.add_argument("--deduplicate", action="store_true")
    args = parser.parse_args()
    raw_records = [x for p in args.inputs for x in read_jsonl(p)]
    if args.schema:
        schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
        if schema.get("type") == "array":
            validate_schema(raw_records, schema)
        else:
            for value in raw_records: validate_schema(value, schema)
    if raw_records and "candidate" not in raw_records[0]:
        report = {"total":len(raw_records), "schema_valid":True}
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(report,ensure_ascii=False,indent=2)); return
    records = [ClassifiedRecord.model_validate(x) for x in raw_records]
    accepted, issues, report = validate_records(records)
    rejected_ids = {issue.record_id for issue in report.rejected}
    non_rejected = [record for record in records if record.candidate.id not in rejected_ids]
    if args.output: write_jsonl(args.output, non_rejected)
    if args.rejected_output: write_jsonl(args.rejected_output, [x.model_dump(mode="json") for x in report.rejected])
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2))


if __name__ == "__main__": main()
