from __future__ import annotations

import re
import unicodedata


PRIVACY_PATTERNS = {
    "PHONE": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "EMAIL": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "ID_CARD": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    "BANK_CARD": re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
}
DANGEROUS_PATTERNS = [re.compile(x, re.I) for x in (r"具体.{0,6}(步骤|方法|剂量)", r"如何.{0,8}(不被发现|实施)", r"哪种.{0,8}(最有效|成功率)")]
META_PATTERNS = [re.compile(x, re.I) for x in (r"正确标签是", r"请把我分类", r"作为\s*(?:AI|人工智能)", r"```")]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(char for char in text if char.isalnum())


def privacy_codes(text: str) -> list[str]:
    return [name for name, pattern in PRIVACY_PATTERNS.items() if pattern.search(text)]


def has_dangerous_detail(text: str) -> bool:
    return any(pattern.search(text) for pattern in DANGEROUS_PATTERNS)


def has_meta_language(text: str) -> bool:
    return any(pattern.search(text) for pattern in META_PATTERNS)


def label_conflicts(record) -> list[str]:
    result = []
    intent, emotion, risk = record.intent.value, record.emotion.value, record.risk.value
    reasons = set(record.reason_codes)
    if emotion == "HIGH_RISK" and risk == "LOW": result.append("HIGH_EMOTION_LOW_RISK")
    if intent == "RISK" and risk == "LOW": result.append("RISK_INTENT_LOW_RISK")
    if risk == "HIGH" and intent == "CHAT": result.append("HIGH_RISK_CHAT")
    if reasons.intersection({"SUICIDE", "SELF_HARM", "HARM_TO_OTHERS"}) and risk != "HIGH": result.append("HARM_REASON_NOT_HIGH")
    if "ACADEMIC_CONTEXT" in reasons and intent == "RISK": result.append("ACADEMIC_CONTEXT_RISK")
    return result

