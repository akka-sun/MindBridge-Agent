from __future__ import annotations

import re
from dataclasses import dataclass

from .models import DocumentSection, KnowledgeChunkRecord, SourceDocument


def tokenize_bm25(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())
    compact = "".join(ch for ch in text.lower() if "\u4e00" <= ch <= "\u9fff")
    return words + [compact[i:i + 2] for i in range(max(0, len(compact) - 1))]


def _record(strategy: str, document: SourceDocument, section: DocumentSection, text: str, index: int, context: str = "") -> KnowledgeChunkRecord:
    retrieval = f"[文档] {document.title}\n[章节] {' / '.join(section.heading_path)}\n"
    if context:
        retrieval += f"[检索背景] {context}\n"
    retrieval += f"[正文] {text}"
    return KnowledgeChunkRecord(chunk_id=f"{strategy}:{document.document_id}:{index:04d}", strategy=strategy, document_id=document.document_id, section_ids=[section.section_id], heading_path=section.heading_path, page_start=section.page_start, page_end=section.page_end, original_text=text, retrieval_text=retrieval, contextual_summary=context, review_status=document.review_status, production_enabled=document.production_enabled)


class FixedChunker:
    def __init__(self, size: int = 512, overlap: int = 64):
        if size <= 0 or overlap < 0 or overlap >= size:
            raise ValueError("size must be positive and overlap must be smaller than size")
        self.size, self.overlap = size, overlap

    def chunk(self, document: SourceDocument) -> list[KnowledgeChunkRecord]:
        output = []
        index = 0
        for section in document.sections:
            start, step = 0, self.size - self.overlap
            while start < len(section.text):
                output.append(_record("fixed", document, section, section.text[start:start + self.size], index))
                index += 1
                start += step
        return output


class SemanticChunker:
    def __init__(self, target_size: int = 500, hard_limit: int = 900):
        self.target_size, self.hard_limit = target_size, hard_limit

    def chunk(self, document: SourceDocument, embedder=None) -> list[KnowledgeChunkRecord]:
        output, index = [], 0
        for section in document.sections:
            units = [x.strip() for x in re.split(r"(?<=[。！？.!?])\s*|\n+", section.text) if x.strip()]
            current = ""
            for unit in units:
                if current and len(current) + len(unit) > self.target_size:
                    output.append(_record("contextual", document, section, current, index))
                    index += 1
                    current = ""
                if len(unit) > self.hard_limit:
                    for start in range(0, len(unit), self.hard_limit):
                        piece = unit[start:start + self.hard_limit]
                        if current:
                            output.append(_record("contextual", document, section, current, index)); index += 1; current = ""
                        output.append(_record("contextual", document, section, piece, index)); index += 1
                else:
                    current += unit
            if current:
                output.append(_record("contextual", document, section, current, index)); index += 1
        return output

