from mindbridge_sft.data_collator import build_assistant_only_labels


def test_only_assistant_tokens_receive_loss():
    ids = [10, 11, 12, 13, 14]
    labels = build_assistant_only_labels(ids, assistant_start=3)
    assert labels == [-100, -100, -100, 13, 14]

