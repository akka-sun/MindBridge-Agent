from app.rag_pipeline.chunking import FixedChunker, SemanticChunker, tokenize_bm25
from app.rag_pipeline.models import DocumentSection, SourceDocument


def doc():
    return SourceDocument(document_id="doc", title="压力管理", source_url="u", local_path="p", organization="WHO", language="zh-CN", region="GLOBAL", license="x", review_status="ok", content_sha256="a" * 64, sections=[DocumentSection(section_id="doc:s1", heading_path=["压力"], text="第一句。第二句。第三句。")])


def test_fixed_chunker_uses_explicit_size_and_overlap():
    chunks = FixedChunker(size=8, overlap=2).chunk(doc())
    assert chunks[0].strategy == "fixed"
    assert chunks[0].original_text == "第一句。第二句。"
    assert chunks[1].original_text.startswith("句。")


def test_semantic_chunker_never_crosses_heading_sections():
    d = doc()
    d.sections.append(DocumentSection(section_id="doc:s2", heading_path=["睡眠"], text="睡眠建议。保持规律。"))
    chunks = SemanticChunker(target_size=100).chunk(d)
    assert all(not ({"压力", "睡眠"} <= set(c.heading_path)) for c in chunks)
    assert {"doc:s1", "doc:s2"} == {c.section_ids[0] for c in chunks}


def test_bm25_tokenizer_keeps_chinese_bigrams_and_words():
    tokens = tokenize_bm25("长期失眠与 anxiety")
    assert "长期" in tokens and "失眠" in tokens and "anxiety" in tokens
