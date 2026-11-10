"""Task 4: correctness and performance of reference vs tiled online-softmax attention.

    python -m benchmarks.bench_attention --out results/bench/attention.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from trialmatch.attention import reference_attention, tiled_attention, torch_sdpa
from trialmatch.modeling import env_info


def err(a, b):
    d = (a.double() - b.double()).abs()
    return float(d.max()), float(d.mean())


def correctness(device):
    rows = []
    g = torch.Generator(device="cpu").manual_seed(0)
    shapes = [  # (B, H, Lq, Lk, D, block)
        (1, 1, 4, 4, 8, 2), (1, 2, 7, 7, 16, 3), (2, 4, 33, 33, 32, 8), (1, 12, 128, 128, 128, 64),
        (1, 12, 200, 200, 128, 64), (2, 3, 1, 50, 64, 16), (1, 4, 5, 37, 32, 8), (1, 12, 300, 300, 128, 128),
    ]
    dtypes = [torch.float64, torch.float32] + ([torch.float16] if device == "cuda" else [])
    for (B, H, Lq, Lk, D, blk) in shapes:
        for causal in (True, False):
            for dt in dtypes:
                q = torch.randn(B, H, Lq, D, generator=g).to(device, dt)
                k = torch.randn(B, H, Lk, D, generator=g).to(device, dt)
                v = torch.randn(B, H, Lk, D, generator=g).to(device, dt)
                gold = reference_attention(q.double(), k.double(), v.double(), causal)  # fp64 ground truth
                ref = reference_attention(q, k, v, causal)
                til = tiled_attention(q, k, v, causal, block_q=blk, block_k=blk)
                sd = torch_sdpa(q.float() if dt == torch.float64 else q, k.float() if dt == torch.float64 else k,
                                v.float() if dt == torch.float64 else v, causal)
                tol = {torch.float64: 1e-10, torch.float32: 1e-5, torch.float16: 5e-3}[dt]
                e_t = err(til, gold)
                e_r = err(ref, gold)
                e_s = err(sd, gold)
                e_tr = err(til, ref)
                rows.append({"shape": [B, H, Lq, Lk, D], "block": blk, "causal": causal, "dtype": str(dt).split(".")[-1],
                             "tiled_vs_fp64_max": e_t[0], "tiled_vs_fp64_mean": e_t[1],
                             "reference_vs_fp64_max": e_r[0], "sdpa_vs_fp64_max": e_s[0],
                             "tiled_vs_reference_max": e_tr[0], "tolerance": tol,
                             "pass": e_t[0] <= tol * max(1.0, float(gold.abs().max())) and bool(torch.isfinite(til).all())})
    # all-masked-row guard / extreme logits (numerical stability)
    q = torch.randn(1, 1, 16, 64).to(device) * 1e3
    k = torch.randn(1, 1, 16, 64).to(device) * 1e3
    v = torch.randn(1, 1, 16, 64).to(device)
    til = tiled_attention(q, k, v, True, block_q=4, block_k=4)
    ref = reference_attention(q.double(), k.double(), v.double(), True)
    rows.append({"shape": [1, 1, 16, 16, 64], "block": 4, "causal": True, "dtype": "float32-largelogits",
                 "tiled_vs_fp64_max": err(til, ref)[0], "tiled_vs_fp64_mean": err(til, ref)[1],
                 "pass": bool(torch.isfinite(til).all()) and err(til, ref)[0] < 1e-3, "tolerance": 1e-3})
    return rows


def bench_one(fn, q, k, v, reps, device):
    for _ in range(2):
        fn(q, k, v)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn(q, k, v)
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    ts.sort()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20 if device == "cuda" else None
    return {"median_ms": 1e3 * ts[len(ts) // 2], "min_ms": 1e3 * ts[0], "extra_peak_mem_mb": peak}


def performance(device, seqs, reps):
    rows = []
    B, H, D = 1, 12, 128  # Qwen2.5-1.5B: 12 query heads, head_dim 128
    dt = torch.float16 if device == "cuda" else torch.float32
    for L in seqs:
        q, k, v = (torch.randn(B, H, L, D, device=device, dtype=dt) for _ in range(3))
        impls = {
            "reference(materialised)": lambda a, b, c: reference_attention(a, b, c, True),
            "tiled(block=64)": lambda a, b, c: tiled_attention(a, b, c, True, block_q=64, block_k=64),
            "tiled(block=128)": lambda a, b, c: tiled_attention(a, b, c, True, block_q=128, block_k=128),
            "torch_sdpa(fused)": lambda a, b, c: torch_sdpa(a, b, c, True),
        }
        for name, fn in impls.items():
            try:
                r = bench_one(fn, q, k, v, reps, device)
                r.update(impl=name, seq_len=L, dtype=str(dt).split(".")[-1])
                rows.append(r)
                print(f"L={L:5d} {name:26s} {r['median_ms']:9.2f} ms  mem {r['extra_peak_mem_mb']}")
            except torch.cuda.OutOfMemoryError:
                rows.append({"impl": name, "seq_len": L, "oom": True})
                torch.cuda.empty_cache()
    # which SDPA backends are available
    backends = {}
    if device == "cuda":
        b = torch.backends.cuda
        for nm in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled"):
            if hasattr(b, nm):
                backends[nm] = bool(getattr(b, nm)())
    return rows, backends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bench/attention.json")
    ap.add_argument("--seqs", default="256,512,1024,2048,4096")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corr = correctness(device)
    n_pass = sum(r["pass"] for r in corr)
    print(f"correctness: {n_pass}/{len(corr)} cases pass")
    for r in corr:
        if not r["pass"]:
            print("FAIL", r)
    perf, backends = performance(device, [int(s) for s in args.seqs.split(",")], args.reps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"env": env_info(), "correctness": corr, "n_pass": n_pass, "n_cases": len(corr),
                               "performance": perf, "sdpa_backends": backends}, indent=2))


if __name__ == "__main__":
    main()
