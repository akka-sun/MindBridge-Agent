from __future__ import annotations

from typing import Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


LABELS = {"intent":["CHAT","CONSULT","RISK"], "emotion":["NORMAL","ANXIETY","DEPRESSED","HIGH_RISK"], "risk":["LOW","MEDIUM","HIGH"]}


def _field_metrics(gold: list[str], pred: list[str], labels: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    macro = sum(f1) / len(labels)
    return {"accuracy":float(accuracy_score(gold,pred)), "macro_f1":float(macro),
            "per_class":{label:{"precision":float(precision[i]),"recall":float(recall[i]),"f1":float(f1[i]),"support":int(support[i])} for i,label in enumerate(labels)},
            "confusion_matrix":confusion_matrix(gold,pred,labels=labels).tolist()}


def evaluate_predictions(gold: Sequence[dict], predictions: Sequence[dict]) -> dict:
    if len(gold) != len(predictions): raise ValueError("gold and predictions must have equal length")
    report = {field:_field_metrics([x[field] for x in gold],[x.get(field,"UNKNOWN") for x in predictions],labels) for field,labels in LABELS.items()}
    total=max(len(gold),1); report["joint_exact_match"] = sum(g==p for g,p in zip(gold,predictions))/total
    report["intent_risk_exact_match"] = sum(g["intent"]==p.get("intent") and g["risk"]==p.get("risk") for g,p in zip(gold,predictions))/total
    high_total=sum(g["risk"]=="HIGH" for g in gold); high_missed=sum(g["risk"]=="HIGH" and p.get("risk")!="HIGH" for g,p in zip(gold,predictions))
    report["risk"]["high_false_negative_rate"] = high_missed/max(high_total,1)
    report["json_valid_rate"] = sum(all(p.get(k) in LABELS[k] for k in LABELS) for p in predictions)/total
    return report

