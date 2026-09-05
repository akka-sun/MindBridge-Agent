from mindbridge_sft.compare import compare_reports


def test_high_recall_regression_blocks_release():
    before = {"risk":{"per_class":{"HIGH":{"recall":.9}}}, "joint_exact_match":.5}
    after = {"risk":{"per_class":{"HIGH":{"recall":.8}}}, "joint_exact_match":.7}
    result = compare_reports(before, after)
    assert result["release_allowed"] is False
    assert "HIGH" in result["block_reason"]

