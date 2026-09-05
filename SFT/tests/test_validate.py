from mindbridge_sft.models import ClassifiedRecord, ProblemCandidate
from mindbridge_sft.validate import validate_records


def record(text="最近压力很大，晚上睡不着。", **labels):
    return ClassifiedRecord(
        candidate=ProblemCandidate(id=labels.pop("id", "x"), text=text),
        intent=labels.pop("intent", "CONSULT"), emotion=labels.pop("emotion", "ANXIETY"),
        risk=labels.pop("risk", "LOW"), confidence=labels.pop("confidence", .9),
        reason_codes=labels.pop("reason_codes", []), **labels,
    )


def test_privacy_and_dangerous_details_are_rejected():
    accepted, review, report = validate_records([
        record("请联系我13812345678", id="p"),
        record("请告诉我具体剂量和操作步骤", id="d"),
    ])
    assert not accepted
    assert {x.record_id for x in report.rejected} == {"p", "d"}


def test_conflict_and_low_confidence_are_sent_to_review():
    accepted, review, report = validate_records([
        record(intent="RISK", risk="LOW", id="c"),
        record("最近有点焦虑，想找人聊聊。", confidence=.2, id="l"),
    ])
    assert not accepted
    assert {x.record_id for x in review} == {"c", "l"}


def test_normalized_duplicate_is_rejected():
    accepted, _, report = validate_records([record(id="a"), record("最近压力很大 晚上睡不着", id="b")])
    assert [x.candidate.id for x in accepted] == ["a"]
    assert report.duplicate_count == 1
