import torch
from typing import List, Tuple, Dict, Any


# ============================================
# Span finder
# ============================================

def _assistant_spans(
    text: str,
    response_part: str = "<|im_start|>assistant\n",
    next_turn_part: str = "<|im_start|>",
    end_part: str = "<|im_end|>",
) -> List[Tuple[int, int]]:
    """Find char-level spans of ALL assistant content (includes <|im_end|>)."""
    spans = []
    pos = 0
    while True:
        i = text.find(response_part, pos)
        if i < 0:
            break
        content_start = i + len(response_part)

        j1 = text.find(end_part, content_start)
        j2 = text.find(next_turn_part, content_start)
        candidates = [j for j in (j1, j2) if j >= 0]
        content_end = min(candidates) if candidates else len(text)
        # include <|im_end|> so model learns to produce the stop token
        if j1 >= 0 and content_end == j1:
            content_end = j1 + len(end_part)

        spans.append((content_start, content_end))
        pos = content_end
    return spans


def _last_assistant_span(text: str) -> List[Tuple[int, int]]:
    """只返回最后一个 assistant 轮的 span"""
    spans = _assistant_spans(text)
    return [spans[-1]] if spans else []


# ============================================
# Token-level label builder
# ============================================

def _build_labels_from_spans(text, tokenizer, spans):
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    offsets = enc["offset_mapping"]

    labels = [-100] * len(input_ids)
    for ti, (s, e) in enumerate(offsets):
        for a, b in spans:
            if s < b and e > a:
                labels[ti] = input_ids[ti]
                break

    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


# ============================================
# Collator: pad input_ids / attention_mask / labels
# ============================================

def _make_collator(tokenizer):
    def collate(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        def pad_1d(seq, pad_value):
            return seq + [pad_value] * (max_len - len(seq))

        return {
            "input_ids":      torch.tensor([pad_1d(f["input_ids"],      pad_id) for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([pad_1d(f["attention_mask"],  0)     for f in features], dtype=torch.long),
            "labels":         torch.tensor([pad_1d(f["labels"],         -100)   for f in features], dtype=torch.long),
        }
    return collate


# ============================================
# Main entry: Mask_Loss
# ============================================

def Mask_Loss(trainer, tokenizer, raw_train_dataset, num_proc=4):
    """
    Train only on the last assistant turn (includes <|im_end|>).
    All other tokens are masked with -100.
    """
    trainer.train_dataset = raw_train_dataset

    def map_fn(ex):
        text = ex["text"]
        spans = _last_assistant_span(text)
        return _build_labels_from_spans(text, tokenizer, spans)

    dataset_tok = trainer.train_dataset.map(
        map_fn,
        num_proc=num_proc,
        remove_columns=trainer.train_dataset.column_names,
    )

    trainer.train_dataset = dataset_tok
    trainer.data_collator = _make_collator(tokenizer)
    trainer.args.remove_unused_columns = False

    n = len(dataset_tok)
    print(f"✅ Mask_Loss applied — {n} samples, training on last assistant turn only")

    return trainer
