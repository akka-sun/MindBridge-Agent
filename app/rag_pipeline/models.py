from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentSection(StrictModel):
    section_id: str = Field(min_length=1)
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    text: str = Field(min_length=1)
    original_text: str | None = None

    @model_validator(mode="after")
    def valid_page_range(self):
        if self.page_start is not None and self.page_end is not None and self.page_end < self.page_start:
            raise ValueError("page_end must not precede page_start")
        return self


class SourceDocument(StrictModel):
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    local_path: str = Field(min_length=1)
    organization: str = Field(min_length=1)
    language: str = Field(min_length=2)
    region: str = Field(min_length=2)
    topics: list[str] = Field(default_factory=list)
    license: str = Field(min_length=1)
    review_status: str = Field(min_length=1)
    production_enabled: bool = False
    content_sha256: str = Field(min_length=64, max_length=64)
    sections: list[DocumentSection] = Field(min_length=1)


class KnowledgeChunkRecord(StrictModel):
    chunk_id: str = Field(min_length=1)
    strategy: Literal["fixed", "contextual"]
    document_id: str = Field(min_length=1)
    section_ids: list[str] = Field(min_length=1)
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    original_text: str = Field(min_length=1)
    retrieval_text: str = Field(min_length=1)
    contextual_summary: str = ""
    review_status: str = Field(min_length=1)
    production_enabled: bool = False
    metadata: dict[str, str | int | float | bool | list[str] | None] = Field(default_factory=dict)


class QueryPlan(StrictModel):
    original_query: str = Field(min_length=1)
    rewritten_queries: list[str] = Field(default_factory=list, max_length=2)
    sub_questions: list[str] = Field(default_factory=list, max_length=3)
    requires_decomposition: bool = False
    preserved_risk_phrases: list[str] = Field(default_factory=list)
    rewrite_labels: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def decomposition_is_consistent(self):
        if self.requires_decomposition and not self.sub_questions:
            raise ValueError("decomposition requires at least one sub-question")
        return self


class RetrievedChunk(StrictModel):
    chunk_id: str
    document_id: str
    section_ids: list[str]
    text: str
    score: float
    ranks: dict[str, int] = Field(default_factory=dict)
    source_url: str = ""


class GoldEvidence(StrictModel):
    document_id: str = Field(min_length=1)
    section_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


QuestionType = Literal[
    "direct", "paraphrase", "multi_condition", "multi_hop", "safety",
    "regional_boundary", "unanswerable", "confusing_negative",
]


class EvaluationCase(StrictModel):
    id: str = Field(pattern=r"^rag-\d{4}$")
    question: str = Field(min_length=2)
    question_type: QuestionType
    difficulty: Literal["easy", "medium", "hard"]
    answerable: bool
    reference_answer: str = Field(min_length=1)
    gold_source_ids: list[str] = Field(default_factory=list)
    gold_evidence: list[GoldEvidence] = Field(default_factory=list)
    required_facts: list[str] = Field(default_factory=list)
    forbidden_facts: list[str] = Field(default_factory=list)
    region: str = "CN"
    risk_level: Literal["normal", "low", "medium", "high"] = "normal"
    generation_method: str = "deepseek-v4-flash"
    review_status: str = "validated"

    @model_validator(mode="after")
    def evidence_matches_answerability(self):
        if not self.answerable and (self.gold_evidence or self.gold_source_ids):
            raise ValueError("unanswerable cases cannot have gold evidence")
        if self.answerable and not self.gold_evidence:
            raise ValueError("answerable cases require gold evidence")
        return self


class ExperimentConfig(StrictModel):
    experiment_id: str = Field(min_length=1)
    index_strategy: Literal["fixed", "contextual"]
    dense_enabled: bool = True
    bm25_enabled: bool = True
    rrf_enabled: bool = True
    reranker_enabled: bool = True
    query_rewrite_enabled: bool = False
    decomposition_enabled: bool = False
    dense_candidates: int = Field(default=30, ge=1)
    bm25_candidates: int = Field(default=30, ge=1)
    rerank_candidates: int = Field(default=30, ge=1, le=80)
    final_top_k: int = Field(default=5, ge=1)
    strict_mode: bool = True

    @model_validator(mode="after")
    def valid_retrieval_graph(self):
        if not self.dense_enabled and not self.bm25_enabled:
            raise ValueError("at least one retriever must be enabled")
        if self.rrf_enabled and not (self.dense_enabled and self.bm25_enabled):
            raise ValueError("RRF requires dense and BM25 retrieval")
        return self

