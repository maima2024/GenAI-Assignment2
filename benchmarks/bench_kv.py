"""Task 3: KV-cached vs uncached greedy decoding (+ scratch decoder with own cache and
pluggable attention, which also validates the tiled attention inside a real model).

    python -m benchmarks.bench_kv --adapter runs/target/adapter --out results/bench/kv.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from trialmatch.data import load_split_frames, row_to_example
from trialmatch.kv_decode import ScratchDecoder, first_divergence, greedy_cached, greedy_uncached
from trialmatch.modeling import TARGET_MODEL, env_info, eos_ids, load_model, load_tokenizer, model_card


def long_ids(tok, frames, n_tokens):
    notes = list(dict.fromkeys(frames["train"]["note"].tolist()))
    text = "Summarize the key clinical findings in the following patient notes.\n\n" + "\n\n".join(notes)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    while len(ids) < n_tokens:
        ids = ids + ids
    return ids[:n_tokens]


def run_cfg(model, ids, n_out, reps, scratch=None):
    res = {}
    fns = {"uncached": lambda: greedy_uncached(model, ids, n_out, ignore_eos=True, record_margins=True),
           "cached": lambda: greedy_cached(model, ids, n_out, ignore_eos=True, record_margins=True)}
    if scratch is not None:
        fns["scratch_cached_sdpa"] = lambda: scratch.greedy(ids, n_out, ignore_eos=True)
    for name, fn in fns.items():
        runs = [fn() for _ in range(reps)]
        tok_lists = [r.tokens for r in runs]
        med = lambda key: statistics.median(r.stats()[key] for r in runs)
        res[name] = {"prefill_s": med("prefill_s"), "decode_s": med("decode_s"), "latency_s": med("latency_s"),
                     "tokens_per_s": med("tokens_per_s"), "decode_tokens_per_s": med("decode_tokens_per_s"),
                     "peak_mem_mb": max(r.stats()["peak_mem_mb"] or 0 for r in runs),
                     "deterministic_across_reps": all(t == tok_lists[0] for t in tok_lists),
                     "tokens": tok_lists[0], "margins": runs[0].margins,
                     "extra": {k: v for k, v in runs[0].extra.items()}}
    ref = res["uncached"]["tokens"]
    for name in res:
        d = first_divergence(ref, res[name]["tokens"])
        res[name]["match_uncached"] = d is None
        res[name]["first_divergence"] = d
        if d is not None and d < len(res["uncached"]["margins"]):
            res[name]["ref_margin_at_divergence"] = res["uncached"]["margins"][d]
    for name in res:  # keep json small
        res[name].pop("margins", None)
        res[name]["tokens"] = res[name]["tokens"][:16]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=TARGET_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--manifest", default="results/split/split_manifest.json")
    ap.add_argument("--out", default="results/bench/kv.json")
    ap.add_argument("--configs", default="128:32,128:128,512:32,512:128,1024:32,1024:128,512:256")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--fp32_check", action="store_true")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    tok = load_tokenizer(args.model)
    model = load_model(args.model, args.adapter, dtype="auto")
    dev = model.device
    frames = load_split_frames(args.manifest)
    scratch = ScratchDecoder(model, attention="sdpa")
    report = {"env": env_info(), "model": args.model, "adapter": args.adapter, "model_card": model_card(model),
              "dtype": str(next(model.parameters()).dtype), "reps": args.reps, "configs": []}

    # warm-up (CUDA context, kernels, allocator)
    w = torch.tensor([long_ids(tok, frames, 64)], device=dev)
    greedy_uncached(model, w, 4, ignore_eos=True)
    greedy_cached(model, w, 4, ignore_eos=True)
    scratch.greedy(w, 4, ignore_eos=True)

    for c in args.configs.split(","):
        n_in, n_out = (int(x) for x in c.split(":"))
        ids = torch.tensor([long_ids(tok, frames, n_in)], device=dev)
        r = run_cfg(model, ids, n_out, args.reps, scratch)
        report["configs"].append({"input_len": n_in, "output_len": n_out, "results": r})
        u, ca = r["uncached"], r["cached"]
        print(f"in={n_in:5d} out={n_out:4d} | uncached {u['latency_s']:.2f}s ({u['tokens_per_s']:.1f} tok/s) "
              f"| cached {ca['latency_s']:.2f}s ({ca['tokens_per_s']:.1f} tok/s, prefill {ca['prefill_s']*1e3:.0f} ms) "
              f"| speedup {u['latency_s']/ca['latency_s']:.2f}x | match={ca['match_uncached']} "
              f"scratch_match={r['scratch_cached_sdpa']['match_uncached']}")

    # ---- scratch decoder correctness on real task prompts, all three attention kernels
    eos = eos_ids(tok)
    exs = [row_to_example(row, tok, 1536) for _, row in frames["val"].head(3).iterrows()]
    sc = []
    for ex in exs:
        ids = tok(ex["prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
        hf_out = model(input_ids=ids, use_cache=False).logits[0, -1].float()
        hf_tokens = greedy_cached(model, ids, 48, eos=eos).tokens
        row = {"annotation_id": ex["annotation_id"], "prompt_tokens": ids.shape[1], "hf_tokens": len(hf_tokens)}
        for attn in ("reference", "tiled", "sdpa"):
            s = ScratchDecoder(model, attention=attn, block=64)
            logit = s.forward(ids, None)[0].float()
            row[f"{attn}_prefill_logit_maxabs_diff"] = float((logit - hf_out).abs().max())
            row[f"{attn}_argmax_equal"] = int(logit.argmax()) == int(hf_out.argmax())
            for use_cache in (True, False):
                toks = s.greedy(ids, 48, eos=eos, use_cache=use_cache).tokens
                key = f"{attn}_{'cached' if use_cache else 'uncached'}"
                row[key + "_match_hf"] = toks == hf_tokens
                row[key + "_first_div"] = first_divergence(hf_tokens, toks)
        sc.append(row)
        print("scratch check:", row)
    report["scratch_checks"] = sc

    if args.fp32_check:  # re-run a subset in fp32 to separate algorithmic from numerical mismatches
        del model, scratch
        torch.cuda.empty_cache()
        m32 = load_model(args.model, args.adapter, dtype="fp32")
        fp = []
        for c in args.configs.split(",")[:4]:
            n_in, n_out = (int(x) for x in c.split(":"))
            ids = torch.tensor([long_ids(tok, frames, n_in)], device=dev)
            a = greedy_uncached(m32, ids, n_out, ignore_eos=True)
            b = greedy_cached(m32, ids, n_out, ignore_eos=True)
            fp.append({"input_len": n_in, "output_len": n_out, "match": a.tokens == b.tokens,
                       "first_divergence": first_divergence(a.tokens, b.tokens),
                       "uncached_s": a.total_s, "cached_s": b.total_s})
            print("fp32:", fp[-1])
        report["fp32_check"] = fp

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
