from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from .chunking import tokenize_bm25
from .models import KnowledgeChunkRecord


class Bm25Index:
    def __init__(self, chunks: list[KnowledgeChunkRecord]):
        self.chunks = chunks
        self.model = BM25Okapi([tokenize_bm25(c.retrieval_text) for c in chunks])

    def search(self, query: str, top_k: int) -> list[tuple[KnowledgeChunkRecord, float]]:
        scores = self.model.get_scores(tokenize_bm25(query))
        ranked = sorted(enumerate(scores), key=lambda pair: (-float(pair[1]), self.chunks[pair[0]].chunk_id))
        return [(self.chunks[i], float(score)) for i, score in ranked[:top_k] if score > 0]


class JsonDenseIndex:
    def __init__(self, chunks: list[KnowledgeChunkRecord], vectors: list[list[float]]):
        self.chunks, self.vectors = chunks, vectors

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[KnowledgeChunkRecord, float]]:
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b)); na = sum(x * x for x in a) ** 0.5; nb = sum(y * y for y in b) ** 0.5
            return dot / (na * nb) if na and nb else 0.0
        ranked = sorted(zip(self.chunks, self.vectors), key=lambda pair: (-cosine(query_vector, pair[1]), pair[0].chunk_id))
        return [(chunk, cosine(query_vector, vector)) for chunk, vector in ranked[:top_k]]


def save_chunks(chunks: list[KnowledgeChunkRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(c.model_dump_json() for c in chunks) + ("\n" if chunks else ""), encoding="utf-8")


def load_chunks(path: Path) -> list[KnowledgeChunkRecord]:
    return [KnowledgeChunkRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_version(chunks: list[KnowledgeChunkRecord], strategy: str, model: str) -> str:
    payload = "".join(c.chunk_id + c.retrieval_text for c in chunks) + strategy + model
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

