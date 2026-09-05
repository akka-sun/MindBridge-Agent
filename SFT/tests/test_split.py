from mindbridge_sft.models import ProblemCandidate, ReviewedRecord
from mindbridge_sft.split import split_records


def item(number, group):
    return ReviewedRecord(candidate=ProblemCandidate(id=str(number),text=f"测试消息内容编号{number}",split_group=group),intent="CHAT",emotion="NORMAL",risk="LOW",review_status="ACCEPT")


def test_split_groups_do_not_cross_boundaries():
    records=[item(i,f"g{i//2}") for i in range(8)]
    result=split_records(records,{"train":4,"valid":2,"test":2},seed=42)
    groups={name:{x.candidate.split_group for x in values} for name,values in result.items()}
    assert groups["train"].isdisjoint(groups["valid"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["valid"].isdisjoint(groups["test"])


def test_insufficient_data_reports_gap():
    try:
        split_records([item(1,"g1")],{"train":2})
    except ValueError as exc:
        assert "gap 1" in str(exc)
    else:
        raise AssertionError("insufficient data must fail")

