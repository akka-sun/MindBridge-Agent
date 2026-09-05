from app.core.enums import IntentType, RiskLevel
from app.services.joint_classifier import JointClassifier


class FakeAi:
    def __init__(self, value): self.value = value
    def complete(self, messages): return self.value


def test_hard_rule_high_cannot_be_downgraded():
    service = JointClassifier(FakeAi('{"intent":"CONSULT","emotion":"ANXIETY","risk":"LOW"}'))
    result = service.classify("我现在想伤害自己", [])
    assert result.intent == IntentType.RISK
    assert result.risk == RiskLevel.HIGH


def test_invalid_json_falls_back_to_rules():
    service = JointClassifier(FakeAi("not json"))
    result = service.classify("最近压力很大", [])
    assert result.intent == IntentType.CONSULT
    assert result.risk == RiskLevel.LOW

