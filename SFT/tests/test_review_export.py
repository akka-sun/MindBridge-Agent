import csv

from mindbridge_sft.models import ClassifiedRecord, ProblemCandidate
from mindbridge_sft.review_export import export_review_csv, import_review_csv


def test_review_csv_round_trip_preserves_chinese(tmp_path):
    record=ClassifiedRecord(candidate=ProblemCandidate(id="x",text="最近考试压力很大。"),intent="CONSULT",emotion="ANXIETY",risk="LOW",confidence=.8)
    path=tmp_path/"review.csv";export_review_csv([record],path)
    with path.open("r",encoding="utf-8-sig",newline="") as handle: rows=list(csv.DictReader(handle))
    rows[0]["review_status"]="ACCEPT";rows[0]["reviewer_ids"]="r01"
    with path.open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    accepted,rejected=import_review_csv([record],path)
    assert accepted[0].candidate.text=="最近考试压力很大。"
    assert not rejected

