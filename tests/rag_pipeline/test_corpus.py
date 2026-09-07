from pathlib import Path

from app.rag_pipeline.corpus import clean_text, extract_html, sha256_file


def test_html_extraction_removes_navigation_and_keeps_content():
    html = "<html><head><title>x</title></head><body><nav>Menu</nav><main><h1>压力</h1><p>保持规律作息。</p></main><footer>Cookie</footer></body></html>"
    sections = extract_html(html, "doc")
    assert sections[0].heading_path == ["压力"]
    assert "Menu" not in sections[0].text and "保持规律作息" in sections[0].text


def test_clean_text_collapses_noise_and_hashes_file(tmp_path: Path):
    assert clean_text("a\n\na\n  b") == "a\nb"
    p = tmp_path / "x.txt"
    p.write_text("abc", encoding="utf-8")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
