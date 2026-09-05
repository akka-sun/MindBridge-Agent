from mindbridge_sft.metrics import evaluate_predictions


def test_metrics_from_hand_checked_predictions():
    gold = [
        {"intent":"CHAT","emotion":"NORMAL","risk":"LOW"},
        {"intent":"CONSULT","emotion":"ANXIETY","risk":"MEDIUM"},
        {"intent":"RISK","emotion":"HIGH_RISK","risk":"HIGH"},
        {"intent":"RISK","emotion":"HIGH_RISK","risk":"HIGH"},
    ]
    pred = [gold[0], gold[1], gold[2], {"intent":"CONSULT","emotion":"DEPRESSED","risk":"MEDIUM"}]
    report = evaluate_predictions(gold, pred)
    assert report["joint_exact_match"] == .75
    assert report["intent"]["accuracy"] == .75
    assert report["intent"]["per_class"]["RISK"]["recall"] == .5
    assert report["risk"]["high_false_negative_rate"] == .5

