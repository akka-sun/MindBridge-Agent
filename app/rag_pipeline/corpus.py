from __future__ import annotations

import hashlib
import re
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from .models import DocumentSection
from .models import SourceDocument


def clean_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in (text or "").splitlines()]
    result, previous = [], None
    for line in lines:
        if not line or line == previous or re.fullmatch(r"\d+", line):
            continue
        result.append(line)
        previous = line
    return "\n".join(result)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_html(html: str, document_id: str) -> list[DocumentSection]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.select("script, style, nav, footer, header, aside, form"):
        tag.decompose()
    main = soup.select_one("main") or soup.body or soup
    sections, heading, buffer, index = [], [], [], 0
    for element in main.find_all(["h1", "h2", "h3", "h4", "p", "li"], recursive=True):
        text = clean_text(element.get_text(" ", strip=True))
        if not text:
            continue
        if element.name.startswith("h"):
            if buffer:
                sections.append(DocumentSection(section_id=f"{document_id}:s{index:04d}", heading_path=heading, text="\n".join(buffer)))
                index += 1; buffer = []
            level = int(element.name[1:])
            heading = heading[:level - 1] + [text]
        else:
            buffer.append(text)
    if buffer:
        sections.append(DocumentSection(section_id=f"{document_id}:s{index:04d}", heading_path=heading, text="\n".join(buffer)))
    return sections


def extract_pdf(data: bytes, document_id: str) -> list[DocumentSection]:
    import io
    reader = PdfReader(io.BytesIO(data))
    sections = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            sections.append(DocumentSection(section_id=f"{document_id}:p{number:04d}", page_start=number, page_end=number, text=text))
    return sections


class CorpusBuilder:
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def build(self, manifest_path: Path, output_path: Path) -> dict:
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        documents, errors = [], []
        for source in manifest.get("sources", []):
            path = self.project_root / "knowledge_sources" / source["local_path"] if not Path(source["local_path"]).is_absolute() else Path(source["local_path"])
            try:
                data = path.read_bytes()
                sections = extract_pdf(data, source["id"]) if path.suffix.lower() == ".pdf" else extract_html(data.decode("utf-8", errors="ignore"), source["id"])
                documents.append(SourceDocument(document_id=source["id"], title=source["title"], source_url=source["source_url"], local_path=source["local_path"], organization=source["organization"], language=source["language"], region=source["region"], topics=source.get("topics", []), license=source["license"], review_status=source["review_status"], production_enabled=False, content_sha256=source["sha256"], sections=sections))
            except Exception as exc:
                errors.append({"source_id": source.get("id"), "error": f"{type(exc).__name__}: {exc}"})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(doc.model_dump_json() for doc in documents) + ("\n" if documents else ""), encoding="utf-8")
        output_path.with_name("build-errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"documents": len(documents), "errors": len(errors), "output": str(output_path)}
