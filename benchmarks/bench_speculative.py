"""Task 5: greedy speculative decoding vs ordinary greedy decoding of the target.

    python -m benchmarks.bench_speculative --target_adapter runs/target/adapter \
        --draft_adapter runs/draft/adapter --out results/bench/speculative.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from trialmatch.data import load_split_frames, row_to_example
from trialmatch.kv_decode import first_divergence, greedy_cached
from trialmatch.modeling import DRAFT_MODEL, TARGET_MODEL, env_info, eos_ids, load_model, load_tokenizer, model_card
from trialmatch.speculative import check_compatible, speculative_greedy


def prompts(tok, frames, n_task):
    out = []
    for _, row in frames["val"].head(n_task).iterrows():  # validation (never test) prompts
        ex = row_to_example(row, tok, 1536)
        out.append(("task", ex["prompt"], 64))
    notes = list(dict.fromkeys(frames["train"]["note"].tolist()))[:3]
    for n in notes:
        msgs = [{"role": "user", "content": "Summarize this patient note in 4 bullet points:\n" + n}]
        out.append(("summary", tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), 128))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TARGET_MODEL)
    ap.add_argument("--target_adapter", default=None)
    ap.add_argument("--draft", default=DRAFT_MODEL)
    ap.add_argument("--draft_adapter", default=None)
    ap.add_argument("--manifest", default="results/split/split_manifest.json")
    ap.add_argument("--ks", default="1,2,4,6")
    ap.add_argument("--n_task", type=int, default=10)
    ap.add_argument("--out", default="results/bench/speculative.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    tok = load_tokenizer(args.target)
    tok_d = load_tokenizer(args.draft)
    target = load_model(args.target, args.target_adapter)
    drafts = {"draft_base": load_model(args.draft, None)}
    if args.draft_adapter:
        drafts["draft_finetuned"] = load_model(args.draft, args.draft_adapter)
    dev = target.device
    eos = eos_ids(tok)
    frames = load_split_frames(args.manifest)
    ps = prompts(tok, frames, args.n_task)
    rep = {"env": env_info(), "target": args.target, "target_adapter": args.target_adapter,
           "draft": args.draft, "draft_adapter": args.draft_adapter,
           "target_card": model_card(target), "draft_card": model_card(drafts["draft_base"]),
           "compatibility": check_compatible(tok, tok_d, target, drafts["draft_base"]),
           "weights_mem_mb": torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else None,
           "runs": []}
    print("compatibility:", rep["compatibility"])
    assert rep["compatibility"]["same_tokenizer_vocab"], "draft/target tokenizers differ"

    enc = [(kind, tok(p, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev), mx) for kind, p, mx in ps]
    # warm-up
    greedy_cached(target, enc[0][1], 8, eos=eos)
    for d in drafts.values():
        speculative_greedy(target, d, enc[0][1], 8, k=2, eos=eos)

    base = []
    for kind, ids, mx in enc:
        r = greedy_cached(target, ids, mx, eos=eos, record_margins=True)
        base.append(r)
    rep["greedy_baseline"] = [{"kind": k, "prompt_tokens": ids.shape[1], **r.stats()}
                              for (k, ids, _), r in zip(enc, base)]

    for dname, dm in drafts.items():
        for k in (int(x) for x in args.ks.split(",")):
            per = []
            for (kind, ids, mx), b in zip(enc, base):
                s = speculative_greedy(target, dm, ids, mx, k=k, eos=eos)
                d = first_divergence(b.tokens, s.tokens)
                per.append({"kind": kind, "prompt_tokens": ids.shape[1], "equal_to_greedy": d is None,
                            "first_divergence": d,
                            "greedy_margin_at_divergence": (b.margins[d] if d is not None and d < len(b.margins) else None),
                            "greedy_latency_s": b.total_s, **s.stats()})
            tot_g = sum(p["greedy_latency_s"] for p in per)
            tot_s = sum(p["latency_s"] for p in per)
            ntok = sum(p["n_tokens"] for p in per)
            summ = {"draft": dname, "k": k, "n_prompts": len(per),
                    "all_equal": all(p["equal_to_greedy"] for p in per),
                    "n_equal": sum(p["equal_to_greedy"] for p in per),
                    "acceptance_rate": sum(p["accepted"] for p in per) / max(1, sum(p["drafted"] for p in per)),
                    "mean_tokens_per_target_call": sum(p["tokens_per_target_call"] for p in per) / len(per),
                    "greedy_total_s": tot_g, "spec_total_s": tot_s, "speedup": tot_g / tot_s,
                    "greedy_tok_s": ntok / tot_g, "spec_tok_s": ntok / tot_s,
                    "peak_mem_mb": max(p["peak_mem_mb"] or 0 for p in per),
                    "by_kind": {}}
            for kind in sorted({p["kind"] for p in per}):
                pk = [p for p in per if p["kind"] == kind]
                summ["by_kind"][kind] = {
                    "acceptance_rate": sum(p["accepted"] for p in pk) / max(1, sum(p["drafted"] for p in pk)),
                    "speedup": sum(p["greedy_latency_s"] for p in pk) / sum(p["latency_s"] for p in pk)}
            rep["runs"].append({"summary": summ, "per_prompt": per})
            print(json.dumps({k2: v for k2, v in summ.items() if k2 != "by_kind"}), summ["by_kind"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
