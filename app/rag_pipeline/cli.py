from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking import FixedChunker, SemanticChunker
from .config import get_rag_settings
from .corpus import CorpusBuilder
from .indexing import save_chunks
from .contextualization import Contextualizer
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
    args = parser.parse_args(argv)
    if args.command == "validate-sources":
        data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        missing = [s["local_path"] for s in data["sources"] if not (Path("knowledge_sources") / s["local_path"]).exists()]
        print(json.dumps({"sources": len(data["sources"]), "missing": missing}, ensure_ascii=False))
        return 1 if missing else 0
    settings = get_rag_settings()
    if args.command == "build-corpus":
        result = CorpusBuilder(settings.project_root).build(Path(args.manifest), Path(args.output)); print(json.dumps(result, ensure_ascii=False)); return 0 if not result["errors"] else 1
    docs = _documents(Path(args.documents))
    chunker = FixedChunker() if args.strategy == "fixed" else SemanticChunker()
    rows = []
    for doc in docs:
        doc_chunks = chunker.chunk(doc)
        if args.strategy == "contextual":
            doc_chunks = Contextualizer().contextualize(doc_chunks, doc.title)
        rows.extend(doc_chunks)
    save_chunks(rows, Path(args.output)); print(json.dumps({"strategy": args.strategy, "chunks": len(rows), "output": args.output}, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
