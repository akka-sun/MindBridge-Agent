from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .clients import JsonFileCache, SiliconFlowClient
from .evaluation import compute_retrieval_metrics
from .indexing import Bm25Index, JsonDenseIndex, load_chunks
from .models import ExperimentConfig
from .retrieval import HybridRetriever


def load_index(path: Path):
    chunks = load_chunks(path / "chunks.jsonl")
    vectors = json.loads((path / "vectors.json").read_text(encoding="utf-8"))
    return JsonDenseIndex(chunks, vectors), Bm25Index(chunks)


def run_experiment(config: ExperimentConfig, index_path: Path, dataset_path: Path, client: SiliconFlowClient, output_root: Path) -> dict:
    dense, bm25 = load_index(index_path)
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    questions = [case["question"] for case in cases]
    vectors = []
    for start in range(0, len(questions), 32):
        vectors.extend(client.embed(questions[start:start + 32]))

    class CachedEmbedder:
        embedding_model = client.embedding_model
        def embed(self, texts):
            return [vectors[questions.index(text)] for text in texts]

    retriever = HybridRetriever(dense, bm25, CachedEmbedder(), client if config.reranker_enabled else None)
    def evaluate_one(case):
        started = time.perf_counter()
        hits = retriever.retrieve(case["question"], top_k=config.final_top_k, dense_candidates=config.dense_candidates, bm25_candidates=config.bm25_candidates, rerank_candidates=config.rerank_candidates, rewrite=config.query_rewrite_enabled, decompose=config.decomposition_enabled, strict=config.strict_mode)
        elapsed = (time.perf_counter() - started) * 1000
        # Compare predictions with expectedChunkIds, never document IDs.
        ids = [item.chunk_id for item in hits]
        # Retrieval quality is measured at chunk granularity.  Keep source-level
        # fields only for analysis; they must not make a neighboring chunk count
        # as a hit.
        expected = set(case.get("expectedChunkIds", []))
        return {"id": case["id"], "question": case["question"], "retrieved": ids, "latency_ms": elapsed, "hit": bool(set(ids) & expected)}, ids, expected
    workers = min(8, max(1, int(__import__("os").environ.get("RAG_EXPERIMENT_WORKERS", "8"))))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        completed = list(pool.map(evaluate_one, cases))
    rows = [item[0] for item in completed]
    predicted = [item[1] for item in completed]
    gold = [item[2] for item in completed]
    metrics = compute_retrieval_metrics(predicted, gold)
    result = {"experiment_id": config.experiment_id, "index_strategy": config.index_strategy, "total_cases": len(cases), "metrics": metrics, "results": rows}
    destination = output_root / config.experiment_id
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
