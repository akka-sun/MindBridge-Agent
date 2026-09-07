from __future__ import annotations

from collections import defaultdict

from .models import QueryPlan, RetrievedChunk


def reciprocal_rank_fusion(rankings: dict[str, list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores = defaultdict(float)
    for ranking in rankings.values():
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class HybridRetriever:
    def __init__(self, dense_index, bm25_index, embedder, reranker=None, query_planner=None):
        self.dense, self.bm25, self.embedder, self.reranker, self.query_planner = dense_index, bm25_index, embedder, reranker, query_planner

    def retrieve(self, query: str, *, top_k: int = 5, dense_candidates: int = 30, bm25_candidates: int = 30, rerank_candidates: int = 30, rewrite: bool = False, decompose: bool = False, strict: bool = True) -> list[RetrievedChunk]:
        plan = self.query_planner.plan(query) if self.query_planner and (rewrite or decompose) else QueryPlan(original_query=query)
        queries = [plan.original_query] + (plan.rewritten_queries if rewrite else []) + (plan.sub_questions if decompose else [])
        records, rankings = {}, {}
        for q_index, current in enumerate(queries):
            dense = self.dense.search(self.embedder.embed([current])[0], dense_candidates)
            sparse = self.bm25.search(current, bm25_candidates)
            for chunk, score in dense + sparse:
                records[chunk.chunk_id] = chunk
            rankings[f"dense:{q_index}"] = [chunk.chunk_id for chunk, _ in dense]
            rankings[f"bm25:{q_index}"] = [chunk.chunk_id for chunk, _ in sparse]
        fused = reciprocal_rank_fusion(rankings)[:rerank_candidates]
        if self.reranker:
            try:
                rows = self.reranker.rerank(query, [records[cid].original_text for cid, _ in fused], min(top_k, len(fused)))
                order = [fused[int(row["index"])] [0] for row in rows if 0 <= int(row.get("index", -1)) < len(fused)]
                scores = dict(fused)
                fused = [(cid, scores[cid]) for cid in order] + [(cid, score) for cid, score in fused if cid not in order]
            except Exception:
                if strict:
                    raise
        result = []
        for cid, score in fused[:top_k]:
            c = records[cid]
            result.append(RetrievedChunk(chunk_id=c.chunk_id, document_id=c.document_id, section_ids=c.section_ids, text=c.original_text, score=score))
        return result
