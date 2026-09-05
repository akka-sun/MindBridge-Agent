from mindbridge_sft.rule_baseline import classify_by_rules


def test_academic_context_does_not_trigger_risk():
    result = classify_by_rules("论文中如何定义自杀风险评估？")
    assert result["intent"] == "CHAT"
    assert result["risk"] == "LOW"


def test_immediate_self_harm_is_high():
    result = classify_by_rules("我现在有伤害自己的冲动，怕控制不住")
    assert result == {"intent":"RISK", "emotion":"HIGH_RISK", "risk":"HIGH"}

