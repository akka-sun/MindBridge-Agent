from __future__ import annotations

from .models import KnowledgeChunkRecord


class Contextualizer:
    def __init__(self, client=None):
        self.client = client

    def contextualize(self, chunks: list[KnowledgeChunkRecord], title: str = "") -> list[KnowledgeChunkRecord]:
        result = []
        for chunk in chunks:
            context = f"该片段来自《{title or chunk.document_id}》，主题为{' / '.join(chunk.heading_path) or '心理健康支持'}，用于检索相关知识。"
            if self.client:
                response = self.client.generate_json("contextualize_chunk", chunk.original_text, {})
                candidate = str(response.get("context", "")).strip()
                if 50 <= len(candidate) <= 240 and response.get("added_claims") is False:
                    context = candidate
            result.append(chunk.model_copy(update={"contextualized_summary": context, "retrieval_text": f"[文档] {title or chunk.document_id}\n[章节] {' / '.join(chunk.heading_path)}\n[检索背景] {context}\n[正文] {chunk.original_text}"}))
        return result

