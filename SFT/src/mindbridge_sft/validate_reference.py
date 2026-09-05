from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .io_utils import read_jsonl
from .quality_rules import has_dangerous_detail, has_meta_language, normalize_text, privacy_codes


def inspect_reference(path: Path) -> dict:
    rows=list(read_jsonl(path)); seen=set();duplicates=privacy=dangerous=meta=empty=0;fields=Counter()
    for row in rows:
        fields.update(row.keys()); text=str(row.get("text") or row.get("input") or row.get("question") or "").strip()
        if not text: empty+=1;continue
        normalized=normalize_text(text);duplicates+=normalized in seen;seen.add(normalized)
        privacy+=bool(privacy_codes(text));dangerous+=has_dangerous_detail(text);meta+=has_meta_language(text)
    return {"path":str(path),"total":len(rows),"empty_text":empty,"duplicates":duplicates,"privacy_hits":privacy,"dangerous_detail_hits":dangerous,"meta_language_hits":meta,"field_counts":dict(fields)}


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--report",required=True);a=p.parse_args();report=inspect_reference(Path(a.input));Path(a.report).parent.mkdir(parents=True,exist_ok=True);Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()

