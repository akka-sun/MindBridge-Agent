from __future__ import annotations

import json
from dataclasses import dataclass

from app.core.enums import EmotionLabel, IntentType, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.ai import AiClient, has_consult_signal, has_high_risk_signal
from app.services.assessment import risk_order


@dataclass(frozen=True)
class JointPrediction:
    intent: IntentType
    emotion: EmotionLabel
    risk: RiskLevel
    source: str


class JointClassifier:
    def __init__(self, ai: AiClient):
        self.ai = ai

    def classify(self, text: str, history: list[AiMessage]) -> JointPrediction:
        hard_high = has_high_risk_signal(text)
        try:
            raw = self.ai.complete(self._prompt(text, history))
            start, end = raw.find("{"), raw.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("joint classifier returned no JSON")
            value = json.loads(raw[start:end + 1])
            if set(value) != {"intent", "emotion", "risk"}:
                raise ValueError("joint classifier returned unexpected fields")
            prediction = JointPrediction(IntentType(value["intent"].upper()), EmotionLabel(value["emotion"].upper()), RiskLevel(value["risk"].upper()), "model")
        except Exception:
            prediction = self._fallback(text)
        if hard_high and (prediction.intent != IntentType.RISK or prediction.risk != RiskLevel.HIGH):
            return JointPrediction(IntentType.RISK, EmotionLabel.HIGH_RISK, RiskLevel.HIGH, "hard_rule")
        if prediction.emotion == EmotionLabel.HIGH_RISK and risk_order(prediction.risk) < risk_order(RiskLevel.HIGH):
            return JointPrediction(IntentType.RISK, EmotionLabel.HIGH_RISK, RiskLevel.HIGH, "safety_merge")
        return prediction

    @staticmethod
    def _prompt(text: str, history: list[AiMessage]) -> list[AiMessage]:
        recent = "无" if not history else "\n".join(f"{x.role}: {x.content}" for x in history[-12:])
        return [
            AiMessage(role="system", content=("你是 MindBridge 联合分类器。只输出严格 JSON，且只能包含 intent、emotion、risk。"
                "intent: CHAT|CONSULT|RISK；emotion: NORMAL|ANXIETY|DEPRESSED|HIGH_RISK；risk: LOW|MEDIUM|HIGH。")),
            AiMessage(role="user", content=f"最近对话：\n{recent}\n\n当前用户消息：\n{text}"),
        ]

    @staticmethod
    def _fallback(text: str) -> JointPrediction:
        if has_high_risk_signal(text):
            return JointPrediction(IntentType.RISK, EmotionLabel.HIGH_RISK, RiskLevel.HIGH, "rule_fallback")
        if has_consult_signal(text):
            emotion = EmotionLabel.DEPRESSED if any(x in text.lower() for x in ("抑郁", "低落", "崩溃", "难过")) else EmotionLabel.ANXIETY
            risk = RiskLevel.MEDIUM if emotion == EmotionLabel.DEPRESSED else RiskLevel.LOW
            return JointPrediction(IntentType.CONSULT, emotion, risk, "rule_fallback")
        return JointPrediction(IntentType.CHAT, EmotionLabel.NORMAL, RiskLevel.LOW, "rule_fallback")

