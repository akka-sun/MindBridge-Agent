QUESTION_QUOTAS = {
    "direct": 90, "paraphrase": 50, "multi_condition": 40, "multi_hop": 35,
    "safety": 35, "regional_boundary": 20, "unanswerable": 20, "confusing_negative": 10,
}


def quota_is_exact(rows: list[dict]) -> bool:
    from collections import Counter
    return len(rows) == 300 and dict(Counter(row.get("question_type") for row in rows)) == QUESTION_QUOTAS

