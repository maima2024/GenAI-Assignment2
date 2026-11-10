"""Generate report figures + a consolidated summary from results/*.json (no GPU needed)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("results")
F = R / "figures"
F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})


def load(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.savefig(F / f"{name}.pdf")
    fig.savefig(F / f"{name}.png")
    plt.close(fig)


summary = {}

# ---------------------------------------------------------------- training curves
logs = sorted(R.glob("train_log_*.json"))
if logs:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for p in logs:
        lg = json.loads(p.read_text())
        name = p.stem.replace("train_log_", "")
        h = lg["log_history"]
        tr = [(e["step"], e["loss"]) for e in h if "loss" in e and "eval_loss" not in e]
        ev = [(e["step"], e["eval_loss"]) for e in h if "eval_loss" in e]
        ax = axes[0] if name == "target" else axes[1]
        if tr:
            ax.plot(*zip(*tr), alpha=0.6, label=f"{name} train")
        if ev:
            ax.plot(*zip(*ev), marker="o", ms=3, label=f"{name} val")
        summary.setdefault("training", {})[name] = {
            "best_val_loss": lg.get("best_val_loss"), "initial_val_loss": lg.get("initial_val_loss"),
            "train_runtime_s": lg.get("train_runtime_s"), "peak_gpu_mem_gb": lg.get("peak_gpu_mem_gb"),
            "trainable_params": lg.get("trainable_params"), "total_params": lg.get("total_params"),
            "args": lg.get("args"), "truncation": lg.get("truncation"),
            "steps_per_epoch": lg.get("steps_per_epoch")}
    axes[0].set_title("Target Qwen2.5-1.5B (LoRA)")
    axes[1].set_title("0.5B sweep (draft candidates)")
    for ax in axes:
        ax.set_xlabel("optimizer step"); ax.set_ylabel("loss"); ax.set_yscale("log"); ax.legend(fontsize=6); ax.grid(alpha=.3)
    save(fig, "training_curves")

# ---------------------------------------------------------------- evaluation
evals = {}
for d in sorted((R / "eval").glob("*")):
    m = d / "metrics.json"
    if m.exists():
        evals[d.name] = json.loads(m.read_text())
summary["eval"] = {}
for name, m in evals.items():
    s = {}
    for g in ("overall", "inclusion", "exclusion"):
        c, e = m[g]["classification"], m[g]["evidence"]
        s[g] = {"n": c["n"], "accuracy": c["accuracy"], "macro_f1": c["macro_f1"], "invalid_rate": c["invalid_rate"],
                "per_class": c["per_class"], "evid_micro_p": e["micro_precision"], "evid_micro_r": e["micro_recall"],
                "evid_micro_f1": e["micro_f1"], "evid_macro_p": e["macro_precision"], "evid_macro_r": e["macro_recall"],
                "evid_macro_f1": e["macro_f1"], "evid_ne_macro_f1": e["nonempty_gold_macro_f1"],
                "empty_agreement": e["empty_agreement_rate"], "gold_empty_n": e["gold_empty_n"]}
    s["json_parse_rate"] = m["json_parse_rate"]
    s["meta"] = {k: m["meta"][k] for k in ("model", "adapter", "split", "n", "decoding", "n_truncated",
                                           "n_gold_evidence_lost_to_truncation", "generation_time_s")}
    summary["eval"][name] = s


def plot_cm(m, group, ax, title):
    cm = m[group]["classification"]["confusion_matrix"]
    M = np.array(cm["matrix"])
    labs = cm["labels"]
    keep = [i for i in range(len(labs)) if M[i].sum() > 0 or M[:, i].sum() > 0]
    M = M[np.ix_(keep, keep)]
    labs = [labs[i] for i in keep]
    short = {"not enough information": "NEI", "not applicable": "N/A", "not included": "not incl.",
             "not excluded": "not excl.", "included": "incl.", "excluded": "excl.", "invalid": "invalid"}
    ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels([short.get(l, l) for l in labs], rotation=45, ha="right")
    ax.set_yticks(range(len(labs))); ax.set_yticklabels([short.get(l, l) for l in labs])
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title(title, fontsize=8)


for tag in ("ft_test", "base_test"):
    if tag in evals:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.0))
        for ax, g in zip(axes, ("overall", "inclusion", "exclusion")):
            plot_cm(evals[tag], g, ax, f"{tag}: {g}")
        save(fig, f"cm_{tag}")

if "base_test" in evals and "ft_test" in evals:
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    groups = ["overall", "inclusion", "exclusion"]
    x = np.arange(3)
    for i, (tag, lab) in enumerate((("base_test", "zero-shot"), ("ft_test", "fine-tuned"))):
        ax.bar(x + i * 0.2 - 0.3, [evals[tag][g]["classification"]["accuracy"] for g in groups], 0.2, label=f"{lab} acc")
        ax.bar(x + i * 0.2 + 0.1, [evals[tag][g]["classification"]["macro_f1"] for g in groups], 0.2, label=f"{lab} macro-F1", hatch="//")
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(0, 1); ax.legend(fontsize=6, ncol=2); ax.grid(axis="y", alpha=.3)
    ax.set_title("Test split: before vs after fine-tuning")
    save(fig, "before_after")

# ---------------------------------------------------------------- KV cache
kv = load("bench/kv.json")
if kv:
    cfgs = kv["configs"]
    labels = [f"{c['input_len']}/{c['output_len']}" for c in cfgs]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    x = np.arange(len(cfgs))
    for i, mode in enumerate(("uncached", "cached", "scratch_cached_sdpa")):
        if mode in cfgs[0]["results"]:
            axes[0].bar(x + (i - 1) * 0.27, [c["results"][mode]["latency_s"] for c in cfgs], 0.27, label=mode)
            axes[1].bar(x + (i - 1) * 0.27, [c["results"][mode]["peak_mem_mb"] for c in cfgs], 0.27, label=mode)
    axes[0].set_yscale("log"); axes[0].set_ylabel("latency (s, log)"); axes[1].set_ylabel("peak GPU memory (MB)")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30); ax.set_xlabel("input/output tokens"); ax.legend(fontsize=6); ax.grid(axis="y", alpha=.3)
    save(fig, "kv_bench")
    summary["kv"] = [{"in": c["input_len"], "out": c["output_len"],
                      **{f"{m}_{k}": c["results"][m][k] for m in c["results"]
                         for k in ("prefill_s", "decode_s", "latency_s", "tokens_per_s", "peak_mem_mb", "match_uncached")}}
                     for c in cfgs]
    summary["kv_scratch_checks"] = kv.get("scratch_checks")
    summary["kv_fp32_check"] = kv.get("fp32_check")

# ---------------------------------------------------------------- attention
at = load("bench/attention.json")
if at:
    perf = [r for r in at["performance"] if not r.get("oom")]
    impls = sorted({r["impl"] for r in perf})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for im in impls:
        rs = sorted([r for r in perf if r["impl"] == im], key=lambda r: r["seq_len"])
        axes[0].plot([r["seq_len"] for r in rs], [r["median_ms"] for r in rs], marker="o", label=im)
        if rs and rs[0].get("extra_peak_mem_mb") is not None:
            axes[1].plot([r["seq_len"] for r in rs], [r["extra_peak_mem_mb"] for r in rs], marker="o", label=im)
    for ax, yl in zip(axes, ("time (ms)", "extra peak memory (MB)")):
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("sequence length"); ax.set_ylabel(yl)
        ax.legend(fontsize=6); ax.grid(alpha=.3)
    save(fig, "attention_bench")
    summary["attention"] = {"n_pass": at["n_pass"], "n_cases": at["n_cases"],
                            "max_err_by_dtype": {}, "perf": perf, "backends": at.get("sdpa_backends")}
    for r in at["correctness"]:
        k = r["dtype"]
        summary["attention"]["max_err_by_dtype"][k] = max(summary["attention"]["max_err_by_dtype"].get(k, 0), r["tiled_vs_fp64_max"])

# ---------------------------------------------------------------- speculative
sp = load("bench/speculative.json")
if sp:
    runs = [r["summary"] for r in sp["runs"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for d in sorted({r["draft"] for r in runs}):
        rs = sorted([r for r in runs if r["draft"] == d], key=lambda r: r["k"])
        axes[0].plot([r["k"] for r in rs], [r["speedup"] for r in rs], marker="o", label=d)
        axes[1].plot([r["k"] for r in rs], [r["acceptance_rate"] for r in rs], marker="o", label=d)
    axes[0].axhline(1, color="gray", ls="--", lw=.8)
    axes[0].set_ylabel("speed-up vs greedy"); axes[1].set_ylabel("draft acceptance rate"); axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("draft length k"); ax.legend(fontsize=6); ax.grid(alpha=.3)
    save(fig, "speculative_bench")
    summary["speculative"] = {"runs": runs, "compatibility": sp["compatibility"],
                              "baseline": sp["greedy_baseline"], "weights_mem_mb": sp.get("weights_mem_mb")}

sel = load("selection.json")
if sel:
    summary["selection"] = sel
(R / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("figures:", sorted(p.name for p in F.glob("*.png")))
