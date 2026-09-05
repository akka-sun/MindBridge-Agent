from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .classify import build_classification_prompt
from .io_utils import read_jsonl, write_jsonl
from .metrics import evaluate_predictions
from .models import ProblemCandidate
from .rule_baseline import classify_by_rules


SYSTEM_PROMPT = "你是 MindBridge 的意图、情绪与安全风险分类器。只输出含 intent、emotion、risk 的 JSON。"


def parse_prediction(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start: raise ValueError("model output contains no JSON object")
    value=json.loads(raw[start:end+1]); expected={"intent","emotion","risk"}
    if set(value)!=expected: raise ValueError("model output fields do not match schema")
    return value


def load_model(model_path: str, adapter: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(model_path,device_map="auto",torch_dtype="auto",trust_remote_code=True)
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,adapter)
    return model,tokenizer


def infer(model, tokenizer, candidate: ProblemCandidate) -> tuple[str,float,int]:
    import torch
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":build_classification_prompt(candidate)}]
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);inputs=tokenizer(text,return_tensors="pt").to(model.device);start=time.perf_counter()
    with torch.inference_mode(): output=model.generate(**inputs,max_new_tokens=96,do_sample=False)
    latency=(time.perf_counter()-start)*1000;generated=output[0,inputs["input_ids"].shape[1]:];raw=tokenizer.decode(generated,skip_special_tokens=True)
    return raw,latency,int(generated.shape[0])


def evaluate_file(test_file: Path, model_path: str | None, adapter: str | None = None, rule_only: bool = False):
    items=list(read_jsonl(test_file)); model=tokenizer=None
    if not rule_only: model,tokenizer=load_model(model_path or "",adapter)
    gold=[];pred=[];cases=[]
    for item in items:
        candidate_data=item.get("candidate",item); candidate=ProblemCandidate.model_validate(candidate_data)
        label=item.get("label") or {k:item[k] for k in ("intent","emotion","risk")}; gold.append(label)
        try:
            if rule_only: prediction=classify_by_rules(candidate.text);raw=json.dumps(prediction,ensure_ascii=False);latency=tokens=0
            else: raw,latency,tokens=infer(model,tokenizer,candidate);prediction=parse_prediction(raw)
            error=None
        except Exception as exc:
            prediction={"intent":"UNKNOWN","emotion":"UNKNOWN","risk":"UNKNOWN"};raw=locals().get("raw","");latency=tokens=0;error=repr(exc)
        pred.append(prediction);cases.append({"id":candidate.id,"gold":label,"prediction":prediction,"raw_output":raw,"error":error,"latency_ms":latency,"output_tokens":tokens,"scene":candidate.scene,"boundary_type":candidate.boundary_type})
    report=evaluate_predictions(gold,pred);report["average_latency_ms"]=sum(x["latency_ms"] for x in cases)/max(len(cases),1);report["average_output_tokens"]=sum(x["output_tokens"] for x in cases)/max(len(cases),1)
    return report,cases


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--model");p.add_argument("--adapter");p.add_argument("--test-file",required=True);p.add_argument("--redteam-file");p.add_argument("--output",required=True);p.add_argument("--predictions");p.add_argument("--rule-only",action="store_true");a=p.parse_args()
    report,cases=evaluate_file(Path(a.test_file),a.model,a.adapter,a.rule_only);Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    if a.predictions:write_jsonl(a.predictions,cases)


if __name__ == "__main__": main()

