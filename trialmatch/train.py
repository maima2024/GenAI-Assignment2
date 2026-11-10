"""Supervised LoRA fine-tuning on the prescribed training split.

Objective: causal-LM cross-entropy on the ANSWER tokens only (prompt tokens masked with -100).
The answer is a compact JSON object {"evidence": [...], "label": "..."} followed by <|im_end|>.

    python -m trialmatch.train --model Qwen/Qwen2.5-1.5B-Instruct --out runs/target
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .data import load_split_frames, row_to_example
from .modeling import env_info, load_tokenizer, _dtype_kwargs


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tok, max_len):
        self.items = []
        end = "<|im_end|>"
        self.n_cut = 0
        for ex in examples:
            p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
            a = tok(ex["target"] + end, add_special_tokens=False)["input_ids"]
            ids = p + a
            labels = [-100] * len(p) + a
            if len(ids) > max_len:  # should not happen: prompt budget already enforced
                self.n_cut += 1
                ids, labels = ids[-max_len:], labels[-max_len:]
            self.items.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collator(pad_id):
    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):  # right padding for training
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            lab[i, :n] = torch.tensor(b["labels"])
            att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--manifest", default="results/split/split_manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_prompt_tokens", type=int, default=1536)
    ap.add_argument("--eval_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(args.model)
    frames = load_split_frames(args.manifest)

    ex = {}
    trunc_report = {}
    for split in ("train", "val"):
        ex[split] = [row_to_example(r, tok, args.max_prompt_tokens) for _, r in frames[split].iterrows()]
        tr = [e["truncation"] for e in ex[split]]
        trunc_report[split] = {
            "n": len(tr), "n_truncated": sum(t["truncated"] for t in tr),
            "n_gold_evidence_lost": sum(bool(t["gold_evidence_lost"]) for t in tr),
            "max_prompt_tokens": max(t["prompt_tokens"] for t in tr),
            "mean_prompt_tokens": sum(t["prompt_tokens"] for t in tr) / len(tr),
            "max_original_prompt_tokens": max(t["original_prompt_tokens"] for t in tr)}
    target_lens = [len(tok(e["target"])["input_ids"]) + 1 for e in ex["train"]]
    trunc_report["train"]["max_target_tokens"] = max(target_lens)
    trunc_report["train"]["mean_target_tokens"] = sum(target_lens) / len(target_lens)
    print("truncation:", json.dumps(trunc_report, indent=2))

    max_len = args.max_prompt_tokens + 64
    ds_train = SFTDataset(ex["train"], tok, max_len)
    ds_val = SFTDataset(ex["val"], tok, max_len)

    # load base weights in fp32 for numerically safe fp16 AMP on T4 (no bf16 support)
    load_dt = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="sdpa", **_dtype_kwargs(load_dt))
    model.config.use_cache = False
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        weight_decay=0.0,
        logging_steps=2,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],  # PeftModel hides the forward signature; needed for eval_loss
        seed=args.seed,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val,
                      data_collator=make_collator(tok.pad_token_id))
    t0 = time.time()
    init_eval = trainer.evaluate()
    train_out = trainer.train()
    train_time = time.time() - t0
    final_eval = trainer.evaluate()
    trainer.model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")

    peak_mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    log = {
        "args": vars(args), "env": env_info(), "trainable_params": trainable, "total_params": total,
        "n_train_examples": len(ds_train), "n_val_examples": len(ds_val),
        "effective_batch": args.batch * args.grad_accum,
        "steps_per_epoch": math.ceil(len(ds_train) / (args.batch * args.grad_accum)),
        "initial_val_loss": init_eval.get("eval_loss"), "best_val_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint, "final_val_loss_best_model": final_eval.get("eval_loss"),
        "train_runtime_s": train_time, "train_metrics": train_out.metrics, "peak_gpu_mem_gb": peak_mem,
        "truncation": trunc_report, "log_history": trainer.state.log_history,
    }
    (out / "train_log.json").write_text(json.dumps(log, indent=2, default=str))
    print("done; best val loss", trainer.state.best_metric, "time", round(train_time, 1), "s")


if __name__ == "__main__":
    main()
