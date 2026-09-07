from app.rag_pipeline.dataset import quota_counts, validate_evidence_quote
from app.rag_pipeline.evaluation import compute_retrieval_metrics


def test_quota_counts_reports_question_types():
    rows = [{"question_type": "direct"}, {"question_type": "direct"}, {"question_type": "safety"}]
    assert quota_counts(rows) == {"direct": 2, "safety": 1}


def test_evidence_quote_must_exist_in_section_text():
    assert validate_evidence_quote("保持规律作息", "建议保持规律作息。")
    assert not validate_evidence_quote("不存在", "建议保持规律作息。")


def test_retrieval_metrics_hand_calculated():
    assert compute_retrieval_metrics([["a", "b"], ["c"]], [{"a"}, {"x"}]) == {"hit_rate": 0.5, "mrr": 0.5, "recall": 0.5, "precision": 0.25}
