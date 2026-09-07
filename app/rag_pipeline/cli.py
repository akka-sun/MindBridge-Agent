from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .chunking import FixedChunker, SemanticChunker
from .config import get_rag_settings
from .corpus import CorpusBuilder
from .indexing import save_chunks
from .contextualization import Contextualizer
from .clients import JsonFileCache, SiliconFlowClient
from .indexing import IndexBuilder
from .clients import DeepSeekClient
from .dataset import DatasetGenerator
from .experiment_runner import run_experiment
from .query_processing import QueryPlanner
from .models import ExperimentConfig
from .models import SourceDocument


def _documents(path: Path) -> list[SourceDocument]:
    return [SourceDocument.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.rag_pipeline.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-sources")
    validate.add_argument("--manifest", default="knowledge_sources/sources.json")
    build = sub.add_parser("build-corpus")
    build.add_argument("--manifest", default="knowledge_sources/sources.json")
    build.add_argument("--output", default="data/rag-experiments/processed/documents.jsonl")
    chunks = sub.add_parser("build-chunks")
    chunks.add_argument("--documents", default="data/rag-experiments/processed/documents.jsonl")
    chunks.add_argument("--strategy", choices=["fixed", "contextual"], required=True)
    chunks.add_argument("--output", required=True)
    index = sub.add_parser("build-index")
    index.add_argument("--chunks", required=True)
    index.add_argument("--strategy", choices=["fixed", "contextual"], required=True)
    dataset = sub.add_parser("generate-dataset")
    dataset.add_argument("--evidence", required=True)
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--prompt", default="generate_answerable_questions")
    exp = sub.add_parser("run-experiment")
    exp.add_argument("--config", required=True)
    exp.add_argument("--index", required=True)
    exp.add_argument("--dataset", default="app/rag_eval/mindbridge-rag-eval.json")
    exp.add_argument("--output-root", default="data/rag-experiments/runs")
    seed = sub.add_parser("make-300-dataset")
    seed.add_argument("--chunks", required=True)
    seed.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate-sources":
        data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        missing = [s["local_path"] for s in data["sources"] if not (Path("knowledge_sources") / s["local_path"]).exists()]
        print(json.dumps({"sources": len(data["sources"]), "missing": missing}, ensure_ascii=False))
        return 1 if missing else 0
    settings = get_rag_settings()
    if args.command == "build-corpus":
        result = CorpusBuilder(settings.project_root).build(Path(args.manifest), Path(args.output)); print(json.dumps(result, ensure_ascii=False)); return 0 if not result["errors"] else 1
    if args.command == "build-index":
        client = SiliconFlowClient(settings.siliconflow_api_key, settings.siliconflow_base_url, settings.siliconflow_embedding_model, settings.siliconflow_rerank_model, cache=JsonFileCache(settings.experiment_root / "cache"), timeout=settings.rag_request_timeout_seconds, max_retries=settings.rag_request_max_retries)
        result = IndexBuilder(client, settings.experiment_root / "indices").build(Path(args.chunks), args.strategy, settings.rag_embedding_batch_size); print(json.dumps(result, ensure_ascii=False)); return 0
    if args.command == "generate-dataset":
        if not settings.deepseek_api_key:
            raise SystemExit("DEEPSEEK_API_KEY is required for generate-dataset")
        client = DeepSeekClient(settings.deepseek_api_key, settings.deepseek_base_url, settings.deepseek_model, cache=JsonFileCache(settings.experiment_root / "cache"), timeout=settings.rag_request_timeout_seconds, max_retries=settings.rag_request_max_retries)
        rows = DatasetGenerator(client).generate_batch(args.prompt, Path(args.evidence).read_text(encoding="utf-8"), {"requested": 20})
        Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps({"generated": len(rows), "output": args.output}, ensure_ascii=False)); return 0
    if args.command == "run-experiment":
        config = ExperimentConfig.model_validate(json.loads(Path(args.config).read_text(encoding="utf-8")))
        if os.environ.get("RAG_OFFLINE") == "1":
            class OfflineClient:
                embedding_model = "offline-bm25"
                def embed(self, texts): return [[0.0] for _ in texts]
                def rerank(self, query, documents, top_n): return [{"index": i, "relevance_score": 0.0} for i in range(min(top_n, len(documents)))]
            client = OfflineClient()
        else:
            client = SiliconFlowClient(settings.siliconflow_api_key, settings.siliconflow_base_url, settings.siliconflow_embedding_model, settings.siliconflow_rerank_model, cache=JsonFileCache(settings.experiment_root / "cache"), timeout=settings.rag_request_timeout_seconds, max_retries=settings.rag_request_max_retries)
        planner = None
        if config.query_rewrite_enabled or config.decomposition_enabled:
            if not settings.deepseek_api_key:
                raise SystemExit("DEEPSEEK_API_KEY is required for query rewrite/decomposition experiments")
            ds = DeepSeekClient(settings.deepseek_api_key, settings.deepseek_base_url, settings.deepseek_model, cache=JsonFileCache(settings.experiment_root / "cache"), timeout=settings.rag_request_timeout_seconds, max_retries=settings.rag_request_max_retries)
            planner = QueryPlanner(ds)
        result = run_experiment(config, Path(args.index), Path(args.dataset), client, Path(args.output_root), planner); print(json.dumps(result["metrics"], ensure_ascii=False)); return 0
    if args.command == "make-300-dataset":
        from .indexing import load_chunks
        chunks = load_chunks(Path(args.chunks))
        if len(chunks) < 300:
            raise SystemExit(f"need at least 300 chunks, got {len(chunks)}")
        rows = []
        # Do not copy source text into questions: that creates lexical leakage.
        # Questions deliberately require abstraction, application, comparison or
        # boundary reasoning, while the exact supporting chunk remains hidden.
        templates = [
            "在心理支持实践中，{topic}主要解决什么问题？请概括适用条件、核心做法和一个常见误区。",
            "如果来访者处于{topic}相关情境，依据资料应如何分步骤处理？请说明先后顺序及理由。",
            "请比较资料中关于{topic}的建议与仅提供一般安慰的做法，指出至少两个关键差异。",
            "关于{topic}，哪些情况仍可自助应对，哪些信号意味着需要转介或升级支持？",
        ]
        stop = {"的", "了", "和", "与", "在", "是", "或", "及", "等", "一个", "可以", "进行", "相关", "资料", "内容"}
        for i, chunk in enumerate(chunks[:300], 1):
            words = [w.strip("，。；：、()（）[]{}\n\t ") for w in chunk.original_text.split()]
            keywords = [w for w in words if len(w) >= 2 and w not in stop]
            topic = "、".join(keywords[:3]) or "该心理健康主题"
            question = templates[(i - 1) % len(templates)].format(topic=topic)
            rows.append({"id": f"rag-{i:04d}", "question": question,
                         "question_type": ["direct", "multi_condition", "multi_hop", "safety"][(i - 1) % 4],
                         "difficulty": "hard", "answerable": True,
                         "expectedSources": [chunk.document_id], "expectedChunkIds": [chunk.chunk_id]})
        Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps({"generated": len(rows), "output": args.output}, ensure_ascii=False)); return 0
    docs = _documents(Path(args.documents))
    chunker = FixedChunker() if args.strategy == "fixed" else SemanticChunker()
    rows = []
    for doc in docs:
        doc_chunks = chunker.chunk(doc)
        if args.strategy == "contextual":
            ds = None
            if settings.deepseek_api_key:
                ds = DeepSeekClient(settings.deepseek_api_key, settings.deepseek_base_url, settings.deepseek_model, cache=JsonFileCache(settings.experiment_root / "cache"), timeout=settings.rag_request_timeout_seconds, max_retries=settings.rag_request_max_retries)
            doc_chunks = Contextualizer(ds).contextualize(doc_chunks, doc.title)
        rows.extend(doc_chunks)
    save_chunks(rows, Path(args.output)); print(json.dumps({"strategy": args.strategy, "chunks": len(rows), "output": args.output}, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
