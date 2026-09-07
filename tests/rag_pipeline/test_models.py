import pytest
from pydantic import ValidationError

from app.rag_pipeline.models import (
    DocumentSection,
    EvaluationCase,
    ExperimentConfig,
    GoldEvidence,
    QueryPlan,
)


def test_document_section_rejects_reversed_page_range():
    with pytest.raises(ValidationError):
        DocumentSection(section_id="doc:s1", page_start=3, page_end=2, text="正文")


def test_gold_evidence_requires_literal_quote():
    with pytest.raises(ValidationError):
        GoldEvidence(document_id="doc", section_id="doc:s1", quote="")


def test_query_plan_limits_generated_queries():
    with pytest.raises(ValidationError):
        QueryPlan(
            original_query="我很焦虑",
            rewritten_queries=["a", "b", "c"],
            sub_questions=[],
            requires_decomposition=False,
        )


def test_unanswerable_case_cannot_have_gold_evidence():
    with pytest.raises(ValidationError):
        EvaluationCase(
            id="rag-0001",
            question="知识库没有的问题",
            question_type="unanswerable",
            difficulty="hard",
            answerable=False,
            reference_answer="知识库证据不足",
            gold_source_ids=["doc"],
            gold_evidence=[GoldEvidence(document_id="doc", section_id="doc:s1", quote="证据")],
        )


def test_experiment_rejects_rrf_without_both_retrievers():
    with pytest.raises(ValidationError):
        ExperimentConfig(
            experiment_id="bad",
            index_strategy="fixed",
            dense_enabled=True,
            bm25_enabled=False,
            rrf_enabled=True,
            reranker_enabled=False,
        )
