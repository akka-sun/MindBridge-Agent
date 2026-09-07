from __future__ import annotations

from collections import defaultdict


def compute_retrieval_metrics(predicted: list[list[str]], gold: list[set[str]]) -> dict[str, float]:
    if not predicted:
        return {"hit_rate": 0.0, "mrr": 0.0, "recall": 0.0, "precision": 0.0}
    hit = mrr = recall = precision = 0.0
    for rows, expected in zip(predicted, gold):
        found = [i for i, value in enumerate(rows, 1) if value in expected]
        hit += bool(found)
        mrr += 1 / found[0] if found else 0
        recall += len(set(rows) & expected) / max(1, len(expected))
        precision += len(set(rows) & expected) / max(1, len(rows))
    n = len(predicted)
    return {"hit_rate": hit / n, "mrr": mrr / n, "recall": recall / n, "precision": precision / n}

