from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Intent(str, Enum):
    CHAT = "CHAT"
    CONSULT = "CONSULT"
    RISK = "RISK"


class Emotion(str, Enum):
    NORMAL = "NORMAL"
    ANXIETY = "ANXIETY"
    DEPRESSED = "DEPRESSED"
    HIGH_RISK = "HIGH_RISK"


class Risk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=240)


class JointLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Intent
    emotion: Emotion
    risk: Risk


class ProblemCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    text: str = Field(min_length=1, max_length=320)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=6)
    scene: str | None = None
    language_style: str | None = None
    boundary_type: str | None = None
    ambiguity_note: str | None = None
    generation_batch_id: str | None = None
    prompt_version: str | None = None
    generator_model: str | None = None
    generation_parameters: dict[str, Any] = Field(default_factory=dict)
    split_group: str | None = None
    is_redteam: bool = False

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text cannot be blank")
        return value


class ClassifiedRecord(JointLabel):
    candidate: ProblemCandidate
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason_codes: list[str] = Field(default_factory=list)
    classifier_model: str | None = None
    classifier_prompt_version: str | None = None
    request_id: str | None = None


class ReviewedRecord(ClassifiedRecord):
    review_status: str = "PENDING"
    reviewer_ids: list[str] = Field(default_factory=list)
    reviewed_at: datetime | None = None
    review_note: str = ""
    data_version: str = "intent-risk-v1.0"

    def mark_reviewed(self) -> "ReviewedRecord":
        if self.reviewed_at is None:
            self.reviewed_at = datetime.now(timezone.utc)
        return self


class ReviewIssue(BaseModel):
    record_id: str
    code: str
    severity: Literal["REJECT", "REVIEW"]
    message: str


class QualityReport(BaseModel):
    total: int
    accepted: int
    needs_review: int
    rejected_count: int
    duplicate_count: int = 0
    rejected: list[ReviewIssue] = Field(default_factory=list)
    review_issues: list[ReviewIssue] = Field(default_factory=list)
    distributions: dict[str, dict[str, int]] = Field(default_factory=dict)


class GenerationReport(BaseModel):
    requested: int
    generated: int
    skipped_batches: int = 0
    failed_batches: int = 0


class ClassificationReport(BaseModel):
    total: int
    classified: int
    failed: int

