"""Criterion-level evaluation on the fixed validation or test split.

    python -m trialmatch.evaluate --model Qwen/Qwen2.5-1.5B-Instruct --adapter runs/target/adapter \
        --split test --out results/eval/ft_test
Decoding: deterministic greedy, max_new_tokens=64, repetition_penalty=1.0, batched with left padding.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import load_split_frames, row_to_example
from .labels import parse_output
from .metrics import full_report
from .modeling import env_info, eos_ids, load_model, load_tokenizer, model_card
from .predict import hf_generate_batch, result_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--manifest", default="results/split/split_manifest.json")
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--max_prompt_tokens", type=int, default=1536)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(args.model)
    model = load_model(args.model, args.adapter, dtype=args.dtype)
    eos = eos_ids(tok)
    df = load_split_frames(args.manifest)[args.split]
    exs = [row_to_example(r, tok, args.max_prompt_tokens) for _, r in df.iterrows()]
    if args.limit:
        exs = exs[: args.limit]
    # sort by length for efficient batching, restore order afterwards
    order = sorted(range(len(exs)), key=lambda i: len(exs[i]["prompt"]))
    texts = [None] * len(exs)
    t0 = time.time()
    for b in range(0, len(order), args.batch_size):
        idx = order[b:b + args.batch_size]
        outs = hf_generate_batch(model, tok, [exs[i]["prompt"] for i in idx], args.max_new_tokens, eos)
        for i, t in zip(idx, outs):
            texts[i] = t
        print(f"\r{min(b + args.batch_size, len(order))}/{len(order)}", end="", flush=True)
    elapsed = time.time() - t0
    print()

    records = []
    with open(out / "predictions.jsonl", "w") as f:
        for ex, text in zip(exs, texts):
            kept = ex["truncation"]["kept_sentence_ids"]
            parsed = parse_output(text, ex["criterion_type"], kept)
            rec = result_record(ex["criterion_type"], ex["criterion_text"], parsed, ex["sentences"])
            rec.update({"annotation_id": ex["annotation_id"], "patient_id": ex["patient_id"],
                        "trial_id": ex["trial_id"], "gold_label": ex["gold_label"],
                        "gold_evidence": ex["gold_evidence"], "pred_label": parsed["label"],
                        "pred_evidence": parsed["evidence"], "truncation": ex["truncation"],
                        "sentences": ex["sentences"]})
            records.append(rec)
            f.write(json.dumps(rec) + "\n")
    rep = full_report(records)
    rep["meta"] = {"model": args.model, "adapter": args.adapter, "split": args.split, "n": len(records),
                   "decoding": {"strategy": "greedy", "do_sample": False, "num_beams": 1,
                                "max_new_tokens": args.max_new_tokens, "repetition_penalty": 1.0,
                                "batch_size": args.batch_size, "padding": "left",
                                "dtype": str(next(model.parameters()).dtype)},
                   "max_prompt_tokens": args.max_prompt_tokens,
                   "n_truncated": sum(e["truncation"]["truncated"] for e in exs),
                   "n_gold_evidence_lost_to_truncation": sum(bool(e["truncation"]["gold_evidence_lost"]) for e in exs),
                   "generation_time_s": elapsed, "model_card": model_card(model), "env": env_info()}
    (out / "metrics.json").write_text(json.dumps(rep, indent=2))
    o = rep["overall"]
    print(json.dumps({"acc": o["classification"]["accuracy"], "macro_f1": o["classification"]["macro_f1"],
                      "evid_micro_f1": o["evidence"]["micro_f1"], "evid_macro_f1": o["evidence"]["macro_f1"],
                      "json_parse_rate": rep["json_parse_rate"]}, indent=2))
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
