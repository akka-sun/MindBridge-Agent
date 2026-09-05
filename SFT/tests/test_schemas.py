import json
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate


ROOT = Path(__file__).resolve().parents[1]


def schema(name):
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


def test_joint_schema_rejects_extra_fields():
    value = {"intent":"CHAT","emotion":"NORMAL","risk":"LOW","answer":"x"}
    with pytest.raises(ValidationError):
        validate(value, schema("joint_label.schema.json"))


def test_ordinary_and_boundary_examples_validate():
    validate([{"id":"gen-chat-1","text":"帮我解释一下 Java 接口是什么？","scene":"学习","language_style":"口语"}], schema("ordinary_problem.schema.json"))
    validate([{"id":"b-1","text":"我在论文中讨论自杀风险，不是在说自己。","boundary_type":"学术语境","ambiguity_note":"风险词不代表本人风险"}], schema("boundary_problem.schema.json"))

