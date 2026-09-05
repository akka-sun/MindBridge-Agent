from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import yaml

from .data_collator import AssistantOnlyDataCollator
from .io_utils import file_sha256, read_jsonl
from .metrics import evaluate_predictions


def normalize_training_sample(item: dict) -> dict:
    if "messages" in item: return {"messages":item["messages"]}
    candidate=item["candidate"];history=candidate.get("history",[]); history_text="无" if not history else "\n".join(f"{x['role']}: {x['content']}" for x in history)
    user=f"最近对话：\n{history_text}\n\n当前用户消息：\n{candidate['text']}"
    target=json.dumps({k:item[k] for k in ("intent","emotion","risk")},ensure_ascii=False,separators=(",",":"))
    return {"messages":[{"role":"system","content":"你是 MindBridge 的意图、情绪与安全风险分类器。只输出符合格式的 JSON。"},{"role":"user","content":user},{"role":"assistant","content":target}]}


def build_compute_metrics(tokenizer):
    import numpy as np
    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        if isinstance(logits, tuple): logits = logits[0]
        token_ids = np.argmax(logits, axis=-1)
        gold, predictions = [], []
        for predicted_row, label_row in zip(token_ids, labels):
            mask = label_row != -100
            gold_text = tokenizer.decode(label_row[mask], skip_special_tokens=True)
            pred_text = tokenizer.decode(predicted_row[mask], skip_special_tokens=True)
            try: gold_value = json.loads(gold_text[gold_text.find("{"):gold_text.rfind("}")+1])
            except Exception: continue
            try: pred_value = json.loads(pred_text[pred_text.find("{"):pred_text.rfind("}")+1])
            except Exception: pred_value = {"intent":"UNKNOWN","emotion":"UNKNOWN","risk":"UNKNOWN"}
            gold.append(gold_value);predictions.append(pred_value)
        if not gold: return {"intent_macro_f1":0.0,"risk_recall":0.0,"json_valid_rate":0.0}
        report=evaluate_predictions(gold,predictions)
        return {"intent_macro_f1":report["intent"]["macro_f1"],"risk_recall":report["intent"]["per_class"]["RISK"]["recall"],"json_valid_rate":report["json_valid_rate"]}
    return compute_metrics


def train(config_path: Path, train_file: Path, valid_file: Path, output_dir: Path, max_steps: int | None = None) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer
    config=yaml.safe_load(config_path.read_text(encoding="utf-8"));model_cfg=config["model"];train_cfg=config["training"];lora_cfg=config["lora"]
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    tokenizer=AutoTokenizer.from_pretrained(model_cfg["name_or_path"],trust_remote_code=True);tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_cfg["name_or_path"],quantization_config=quant,device_map="auto",trust_remote_code=True);model=prepare_model_for_kbit_training(model)
    peft=LoraConfig(r=lora_cfg["r"],lora_alpha=lora_cfg["alpha"],lora_dropout=lora_cfg["dropout"],target_modules=lora_cfg["target_modules"],task_type="CAUSAL_LM")
    training_args=TrainingArguments(output_dir=str(output_dir),learning_rate=train_cfg["learning_rate"],num_train_epochs=train_cfg["num_train_epochs"],per_device_train_batch_size=train_cfg["per_device_train_batch_size"],per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size",1),gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],bf16=train_cfg.get("bf16",True),logging_steps=train_cfg.get("logging_steps",10),save_strategy="epoch",eval_strategy="epoch",load_best_model_at_end=True,metric_for_best_model="intent_macro_f1",greater_is_better=True,max_steps=max_steps or -1,seed=config.get("seed",42),report_to="none")
    train_ds=Dataset.from_list([normalize_training_sample(x) for x in read_jsonl(train_file)]);valid_ds=Dataset.from_list([normalize_training_sample(x) for x in read_jsonl(valid_file)])
    collator=AssistantOnlyDataCollator(tokenizer,max_length=train_cfg.get("max_seq_length",2048))
    trainer=SFTTrainer(model=model,args=training_args,train_dataset=train_ds,eval_dataset=valid_ds,peft_config=peft,data_collator=collator,processing_class=tokenizer,compute_metrics=build_compute_metrics(tokenizer))
    trainer.train();trainer.save_model(str(output_dir));tokenizer.save_pretrained(str(output_dir))
    manifest={"base_model":model_cfg["name_or_path"],"train_sha256":file_sha256(train_file),"valid_sha256":file_sha256(valid_file),"seed":config.get("seed",42),"config":config,"python":platform.python_version()}
    (output_dir/"training_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--train-file","--train",dest="train",required=True);p.add_argument("--valid-file","--valid",dest="valid",required=True);p.add_argument("--output-dir","--output",dest="output",required=True);p.add_argument("--max-steps",type=int);p.add_argument("--resume-from-checkpoint");p.add_argument("--logging-steps",type=int);p.add_argument("--save-steps",type=int);a=p.parse_args();train(Path(a.config),Path(a.train),Path(a.valid),Path(a.output),a.max_steps)


if __name__ == "__main__": main()
