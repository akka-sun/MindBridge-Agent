from pathlib import Path

from app.rag_pipeline.cli import main


def test_validate_sources_accepts_repository_manifest():
    assert main(["validate-sources", "--manifest", "knowledge_sources/sources.json"]) == 0


def test_build_chunks_writes_jsonl(tmp_path: Path):
    documents = tmp_path / "documents.jsonl"
    documents.write_text('{"document_id":"d","title":"t","source_url":"u","local_path":"p","organization":"o","language":"zh-CN","region":"CN","topics":[],"license":"x","review_status":"ok","production_enabled":false,"content_sha256":"' + 'a' * 64 + '","sections":[{"section_id":"d:s1","heading_path":["h"],"page_start":null,"page_end":null,"text":"保持规律作息。"}]}\n', encoding="utf-8")
    output = tmp_path / "chunks.jsonl"
    assert main(["build-chunks", "--documents", str(documents), "--strategy", "fixed", "--output", str(output)]) == 0
    assert output.exists() and output.read_text(encoding="utf-8").strip()
