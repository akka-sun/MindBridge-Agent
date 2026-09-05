from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare_reports(before: dict, after: dict) -> dict:
    before_high = before["risk"]["per_class"]["HIGH"]["recall"]
    after_high = after["risk"]["per_class"]["HIGH"]["recall"]
    allowed = after_high >= before_high
    return {"release_allowed":allowed, "block_reason":"" if allowed else "HIGH risk recall decreased",
            "deltas":{"joint_exact_match":after.get("joint_exact_match",0)-before.get("joint_exact_match",0),
                      "high_recall":after_high-before_high}}


def render_markdown(before: dict, after: dict) -> str:
    comparison=compare_reports(before,after); decision="允许进入人工灰度审核" if comparison["release_allowed"] else "阻断上线"
    rows=["# SFT 前后评测对比","",f"结论：**{decision}**","","| 指标 | SFT 前 | SFT 后 | 差值 |","|---|---:|---:|---:|",
          f"| 三字段完全匹配率 | {before.get('joint_exact_match',0):.4f} | {after.get('joint_exact_match',0):.4f} | {comparison['deltas']['joint_exact_match']:+.4f} |",
          f"| HIGH Recall | {before['risk']['per_class']['HIGH']['recall']:.4f} | {after['risk']['per_class']['HIGH']['recall']:.4f} | {comparison['deltas']['high_recall']:+.4f} |","",f"阻断原因：{comparison['block_reason'] or '无自动阻断项'}"]
    return "\n".join(rows)+"\n"


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--before","--baseline",dest="before",required=True);p.add_argument("--after","--sft",dest="after",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    before=json.loads(Path(a.before).read_text(encoding="utf-8"));after=json.loads(Path(a.after).read_text(encoding="utf-8"));Path(a.output).write_text(render_markdown(before,after),encoding="utf-8")


if __name__ == "__main__": main()

