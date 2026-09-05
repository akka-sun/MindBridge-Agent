from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def build_assistant_only_labels(input_ids: list[int], assistant_start: int, ignore_index: int = -100) -> list[int]:
    if not 0 <= assistant_start <= len(input_ids): raise ValueError("assistant_start outside token sequence")
    return [ignore_index] * assistant_start + input_ids[assistant_start:]


@dataclass
class AssistantOnlyDataCollator:
    tokenizer: Any
    assistant_marker: str = "<|im_start|>assistant\n"
    max_length: int = 2048

    def __call__(self, examples: list[dict]) -> dict:
        import torch
        texts, starts = [], []
        for example in examples:
            text = self.tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
            marker_index = text.rfind(self.assistant_marker)
            if marker_index < 0: raise ValueError("assistant marker not found in formatted sample")
            prefix = text[:marker_index + len(self.assistant_marker)]
            starts.append(len(self.tokenizer(prefix, add_special_tokens=False)["input_ids"]))
            texts.append(text)
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = batch["input_ids"].clone()
        for row, start in enumerate(starts): labels[row, :start] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

