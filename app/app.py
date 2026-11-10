"""TrialMatch-Qwen: browser-based criterion-level eligibility screening (educational prototype).

Start:   python -m app.app --adapter adapters_export/target_qwen2.5-1.5b_lora \
                           --draft_adapter adapters_export/draft_qwen2.5-0.5b_lora   [--share]
Then open http://127.0.0.1:7860
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
import torch  # noqa: E402

from trialmatch import __version__  # noqa: E402
from trialmatch.aggregate import aggregate  # noqa: E402
from trialmatch.attention import reference_attention, tiled_attention, torch_sdpa  # noqa: E402
from trialmatch.data import build_prompt_with_budget, format_note, parse_note  # noqa: E402
from trialmatch.kv_decode import ScratchDecoder, first_divergence, greedy_cached, greedy_uncached  # noqa: E402
from trialmatch.modeling import (DRAFT_MODEL, TARGET_MODEL, eos_ids, load_model, load_tokenizer,  # noqa: E402
                                 pick_device)
from trialmatch.predict import screen_one  # noqa: E402
from trialmatch.speculative import speculative_greedy  # noqa: E402

DISCLAIMER = ("**Educational screening prototype - NOT a clinical enrollment decision.** "
              "Outputs are produced by a small fine-tuned language model, can be wrong, and must be "
              "verified by qualified clinical research staff. Do not enter identifiable patient data.")
MODES = ["KV-cached greedy (explicit cache loop)", "Speculative greedy (draft + target)",
         "Uncached greedy (reference, slow)", "Scratch decoder (own KV cache + tiled attention)",
         "HF generate (library reference)"]
MAX_NOTE_CHARS, MAX_CRITERIA, MAX_CRIT_CHARS, MAX_NEW = 20000, 40, 1000, 64
REPORT_DIR = ROOT / "app_reports"

FALLBACK_EXAMPLE = {
    "patient_id": "demo-fictional", "trial_id": "DEMO",
    "note": ("A 62-year-old man presents with three months of exertional chest tightness relieved by rest. "
             "He has type 2 diabetes treated with metformin and hypertension. "
             "He smoked one pack per day for 30 years and quit 2 years ago. "
             "An exercise stress test showed reversible inferior ischemia. "
             "Serum creatinine is 1.1 mg/dL. He has no history of stroke or bleeding disorders."),
    "inclusion": ["Age 40 to 80 years", "Stable angina pectoris with documented myocardial ischemia",
                  "Left ventricular ejection fraction of at least 40%"],
    "exclusion": ["Severe renal insufficiency (creatinine > 2.5 mg/dL)", "Prior stroke",
                  "Pregnancy or breastfeeding"],
}


def load_examples():
    ex = [FALLBACK_EXAMPLE]
    p = ROOT / "results" / "split" / "app_examples.json"
    if p.exists():
        try:
            ex += json.loads(p.read_text())  # training-split patients only
        except Exception:
            pass
    return ex


EXAMPLES = load_examples()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ============================================================================ engine
class Engine:
    def __init__(self, args):
        self.args = args
        self.device = pick_device(args.device)
        print(f"[app] loading target {args.target} adapter={args.adapter} on {self.device} ...", flush=True)
        self.tok = load_tokenizer(args.target)
        self.target = load_model(args.target, args.adapter, dtype=args.dtype, device=self.device)
        self.eos = eos_ids(self.tok)
        self._draft = None
        self.scratch = ScratchDecoder(self.target, attention="tiled", block=64)
        print("[app] target ready", flush=True)

    @property
    def draft(self):
        if self._draft is None:
            print(f"[app] loading draft {self.args.draft} adapter={self.args.draft_adapter} ...", flush=True)
            self._draft = load_model(self.args.draft, self.args.draft_adapter, dtype=self.args.dtype, device=self.device)
        return self._draft

    def decoder(self, mode, k=4, max_new=MAX_NEW, ignore_eos=False):
        if mode.startswith("KV-cached"):
            return lambda ids: self._pack(greedy_cached(self.target, ids, max_new, self.eos, ignore_eos))
        if mode.startswith("Speculative"):
            return lambda ids: self._pack(speculative_greedy(self.target, self.draft, ids, max_new, k=k, eos=self.eos, ignore_eos=ignore_eos))
        if mode.startswith("Uncached"):
            return lambda ids: self._pack(greedy_uncached(self.target, ids, max_new, self.eos, ignore_eos))
        if mode.startswith("Scratch"):
            return lambda ids: self._pack(self.scratch.greedy(ids, max_new, self.eos, ignore_eos))
        if mode.startswith("HF generate"):
            return self._hf_generate(max_new, ignore_eos)
        raise ValueError(mode)

    @staticmethod
    def _pack(r):
        return r.tokens, r.stats()

    def _hf_generate(self, max_new, ignore_eos):
        @torch.no_grad()
        def run(ids):
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            out = self.target.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                                       min_new_tokens=max_new if ignore_eos else None,
                                       do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                                       repetition_penalty=1.0, eos_token_id=self.eos, pad_token_id=self.tok.pad_token_id)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            t = time.perf_counter() - t0
            toks = out[0, ids.shape[1]:].tolist()
            pk = torch.cuda.max_memory_allocated() / 2**20 if self.device.startswith("cuda") else None
            return toks, {"mode": "hf-generate", "latency_s": t, "n_tokens": len(toks),
                          "tokens_per_s": len(toks) / t if t else 0, "peak_mem_mb": pk}
        return run


ENGINE: Engine | None = None


# ============================================================================ input handling
def clean_criteria(text: str) -> list[str]:
    out = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        s = re.sub(r"^(?:[\*\-•●]+|\(?\d+[\.\)])\s*", "", s).strip()
        if not s or re.fullmatch(r"(?i)(inclusion|exclusion)\s+criteria\s*:?", s):
            continue
        out.append(s)
    return out


def validate(note, inc_text, exc_text):
    if not note or len(note.strip()) < 20:
        raise gr.Error("Please enter a patient note (at least 20 characters).")
    if len(note) > MAX_NOTE_CHARS:
        raise gr.Error(f"Patient note is too long ({len(note)} chars; limit {MAX_NOTE_CHARS}).")
    inc, exc = clean_criteria(inc_text), clean_criteria(exc_text)
    if not inc and not exc:
        raise gr.Error("Enter at least one inclusion or exclusion criterion (one per line), or fetch a trial by NCT ID.")
    if len(inc) + len(exc) > MAX_CRITERIA:
        raise gr.Error(f"Too many criteria ({len(inc) + len(exc)}); the limit is {MAX_CRITERIA}.")
    long_ = [c for c in inc + exc if len(c) > MAX_CRIT_CHARS]
    if long_:
        raise gr.Error(f"A criterion exceeds {MAX_CRIT_CHARS} characters: '{long_[0][:80]}...'")
    sents = parse_note(note)
    if not sents:
        raise gr.Error("Could not split the note into sentences.")
    return sents, inc, exc


def fetch_trial(nct):
    nct = (nct or "").strip().upper()
    if not re.fullmatch(r"NCT\d{8}", nct):
        raise gr.Error("NCT ID must look like NCT01234567.")
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct}?fields=protocolSection.eligibilityModule,protocolSection.identificationModule"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "trialmatch-edu"}), timeout=15) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        raise gr.Error(f"Could not fetch {nct} from ClinicalTrials.gov ({type(e).__name__}). Paste the criteria manually instead.")
    ps = data.get("protocolSection", {})
    title = ps.get("identificationModule", {}).get("briefTitle", "")
    text = ps.get("eligibilityModule", {}).get("eligibilityCriteria", "")
    if not text:
        raise gr.Error(f"{nct} has no eligibility criteria text.")
    parts = re.split(r"(?i)exclusion criteria\s*:?", text, maxsplit=1)
    inc_part = re.sub(r"(?i)inclusion criteria\s*:?", "", parts[0])
    exc_part = parts[1] if len(parts) > 1 else ""
    inc, exc = clean_criteria(inc_part), clean_criteria(exc_part)
    return "\n".join(inc[:MAX_CRITERIA]), "\n".join(exc[:MAX_CRITERIA]), f"Loaded **{nct}** - {title} ({len(inc)} inclusion / {len(exc)} exclusion criteria)"


def load_example(name):
    ex = next((e for e in EXAMPLES if f"{e['patient_id']} / {e['trial_id']}" == name), EXAMPLES[0])
    return ex["note"], "\n".join(ex["inclusion"]), "\n".join(ex["exclusion"])


# ============================================================================ screening
LABEL_COLORS = {"included": "#1a7f37", "not excluded": "#1a7f37", "excluded": "#cf222e", "not included": "#cf222e",
                "not enough information": "#9a6700", "not applicable": "#57606a", "invalid": "#8250df"}


def note_html(sents, highlight=()):
    rows = []
    for i, s in sents:
        st = "background:#fff3b0;font-weight:600;" if i in highlight else ""
        rows.append(f"<div style='{st}padding:2px 4px;border-radius:3px'><code>{i}</code> {html.escape(s)}</div>")
    return "<div style='font-size:14px;line-height:1.5'>" + "".join(rows) + "</div>"


def run_screen(note, inc_text, exc_text, mode, k, progress=gr.Progress()):
    if ENGINE is None:
        raise gr.Error("Model is not loaded yet - please wait for start-up to finish.")
    sents, inc, exc = validate(note, inc_text, exc_text)
    crit = [("inclusion", c) for c in inc] + [("exclusion", c) for c in exc]
    try:
        dec = ENGINE.decoder(mode, int(k))
    except Exception as e:
        raise gr.Error(f"Could not prepare decoder: {e}")
    results = []
    t0 = time.time()
    for j, (ct, c) in enumerate(crit):
        progress((j, len(crit)), desc=f"criterion {j + 1}/{len(crit)}")
        try:
            r = screen_one(ENGINE.target, ENGINE.tok, sents, ct, c, dec, ENGINE.args.max_prompt_tokens, MAX_NEW)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise gr.Error("GPU out of memory - try a shorter note or the KV-cached mode.")
        except Exception as e:
            raise gr.Error(f"Inference failed on criterion {j + 1}: {type(e).__name__}: {e}")
        r["index"] = j + 1
        results.append(r)
    total = time.time() - t0
    agg = aggregate(results)
    prov = provenance(note, crit, mode, k, total)
    payload = {"provenance": prov, "aggregate": agg, "note_sentences": sents, "results": results}
    files = write_reports(payload)
    table = [[r["index"], r["criterion_type"], r["criterion"], r["label"], ", ".join(map(str, r["evidence_ids"])) or "-",
              " | ".join(r["evidence_text"]) or "-", r["explanation"], ", ".join(r["flags"]) or "-",
              f"{r['stats'].get('latency_s', 0):.2f}"] for r in results]
    trunc = [r for r in results if r["truncation"]["truncated"]]
    md = summary_md(agg, prov, trunc)
    choices = [f"{r['index']}. [{r['criterion_type'][:3]}] {r['criterion'][:70]}" for r in results]
    return (table, md, note_html(sents), files, payload,
            gr.update(choices=choices, value=choices[0] if choices else None))


def summary_md(agg, prov, trunc):
    col = {"R1": "#cf222e", "R2": "#9a6700", "R3": "#1a7f37"}[agg["rule"]]
    lines = [f"### Preliminary outcome: <span style='color:{col}'>{agg['verdict']}</span>  (rule {agg['rule']})",
             "*Aggregated by explicit rules from criterion-level predictions; not a ground-truth trial label.*",
             f"- Criteria evaluated: {agg['n_criteria']}  |  label counts: " + ", ".join(f"{k}: {v}" for k, v in agg["label_counts"].items())]
    if agg["failing_criteria"]:
        lines.append("- **Failing criteria:** " + "; ".join(agg["failing_criteria"]))
    if agg["missing_information"]:
        lines.append("- **Missing information needed for:** " + "; ".join(agg["missing_information"]))
    if agg["absence_based_for_review"]:
        lines.append("- **Decided from absence of documentation (verify):** " + "; ".join(agg["absence_based_for_review"]))
    if trunc:
        lines.append(f"- ⚠ Note truncated for {len(trunc)} criteria (sentences removed: {trunc[0]['truncation']['removed_sentence_ids']})")
    lines.append(f"- Decoding: {prov['decoding_mode']} | total time {prov['total_time_s']:.1f}s | device {prov['device']}")
    return "\n".join(lines)


def provenance(note, crit, mode, k, total):
    a = ENGINE.args
    return {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "app_version": __version__, "git_commit": git_commit(),
            "target_model": a.target, "target_adapter": a.adapter, "draft_model": a.draft if mode.startswith("Spec") else None,
            "draft_adapter": a.draft_adapter if mode.startswith("Spec") else None, "decoding_mode": mode,
            "speculative_k": int(k) if mode.startswith("Spec") else None, "greedy": True, "max_new_tokens": MAX_NEW,
            "device": ENGINE.device, "dtype": str(next(ENGINE.target.parameters()).dtype),
            "note_sha256": hashlib.sha256(note.encode()).hexdigest(),
            "criteria_sha256": hashlib.sha256(json.dumps(crit).encode()).hexdigest(), "total_time_s": total,
            "disclaimer": "Educational screening prototype - not a clinical enrollment decision."}


def write_reports(payload):
    REPORT_DIR.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    jpath = REPORT_DIR / f"screening_{stamp}.json"
    hpath = REPORT_DIR / f"screening_{stamp}.html"
    jpath.write_text(json.dumps(payload, indent=2, default=str))
    p, agg = payload["provenance"], payload["aggregate"]
    rows = "".join(
        f"<tr><td>{r['index']}</td><td>{r['criterion_type']}</td><td>{html.escape(r['criterion'])}</td>"
        f"<td style='color:{LABEL_COLORS.get(r['label'], '#000')};font-weight:600'>{r['label']}</td>"
        f"<td>{', '.join(map(str, r['evidence_ids'])) or '-'}</td>"
        f"<td>{'<br>'.join(html.escape(t) for t in r['evidence_text']) or '-'}</td>"
        f"<td>{html.escape(r['explanation'])}</td><td>{', '.join(r['flags']) or '-'}</td></tr>"
        for r in payload["results"])
    note = "".join(f"<li value='{i}'>{html.escape(s)}</li>" for i, s in payload["note_sentences"])
    prov = "".join(f"<tr><td>{k}</td><td>{html.escape(str(v))}</td></tr>" for k, v in p.items())
    doc = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>Eligibility screening report</title>
<style>body{{font-family:system-ui,Arial;max-width:1100px;margin:24px auto;color:#222}}table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border:1px solid #ccc;padding:5px;vertical-align:top}}th{{background:#f3f3f3}}.warn{{background:#fff4e5;border:1px solid #f0a020;padding:10px;border-radius:6px}}</style></head><body>
<h1>Clinical-trial eligibility screening report</h1>
<div class='warn'><b>Educational screening prototype - NOT a clinical enrollment decision.</b> Criterion labels are model predictions that must be verified by qualified staff.</div>
<h2>Preliminary outcome: {agg['verdict']} (rule {agg['rule']})</h2>
<p>Rule R1: any inclusion 'not included' or exclusion 'excluded' &rarr; likely ineligible; R2: otherwise any 'not enough information' &rarr; undetermined; R3: otherwise potentially eligible.</p>
<p><b>Missing information:</b> {html.escape('; '.join(agg['missing_information']) or 'none')}<br>
<b>Decided from absence of documentation (verify):</b> {html.escape('; '.join(agg['absence_based_for_review']) or 'none')}</p>
<h2>Criterion-level results</h2><table><tr><th>#</th><th>Type</th><th>Criterion</th><th>Label</th><th>Sentence IDs</th><th>Evidence</th><th>Explanation</th><th>Flags</th></tr>{rows}</table>
<h2>Patient note (sentence IDs)</h2><ol start='0'>{note}</ol>
<h2>Provenance</h2><table>{prov}</table></body></html>"""
    hpath.write_text(doc)
    return [str(hpath), str(jpath)]


def inspect(choice, payload):
    if not payload or not choice:
        return "", ""
    idx = int(choice.split(".")[0]) - 1
    r = payload["results"][idx]
    md = (f"**{r['criterion_type'].title()} criterion:** {r['criterion']}\n\n"
          f"**Predicted:** <span style='color:{LABEL_COLORS.get(r['label'], '#000')}'>{r['label']}</span> | "
          f"**cited sentence IDs:** {r['evidence_ids'] or 'none'} | flags: {', '.join(r['flags']) or 'none'}\n\n"
          f"**Explanation:** {r['explanation']}\n\n"
          f"<details><summary>raw model output & decoding stats</summary>\n\n```\n{r['raw_output']}\n```\n"
          f"`{json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r['stats'].items() if k != 'accept_hist'})}`</details>")
    return md, note_html([tuple(s) for s in payload["note_sentences"]], set(r["evidence_ids"]))


# ============================================================================ benchmarks
def run_bench(note, inc_text, exc_text, crit_idx, k, max_new, ignore_eos, progress=gr.Progress()):
    if ENGINE is None:
        raise gr.Error("Model is not loaded yet.")
    sents, inc, exc = validate(note, inc_text, exc_text)
    crit = [("inclusion", c) for c in inc] + [("exclusion", c) for c in exc]
    i = int(crit_idx) - 1
    if not 0 <= i < len(crit):
        raise gr.Error(f"Criterion index must be between 1 and {len(crit)}.")
    ct, c = crit[i]
    prompt, _ = build_prompt_with_budget(ENGINE.tok, sents, ct, c, ENGINE.args.max_prompt_tokens)
    ids = ENGINE.tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(ENGINE.device)
    max_new = int(max_new)
    rows, ref = [], None
    modes = ["Uncached greedy (reference, slow)", "KV-cached greedy (explicit cache loop)",
             "Scratch decoder (own KV cache + tiled attention)", "Speculative greedy (draft + target)", "HF generate (library reference)"]
    ENGINE.decoder(modes[1], int(k), 4, True)(ids)  # warm-up
    for j, m in enumerate(modes):
        progress((j, len(modes)), desc=m)
        try:
            toks, st = ENGINE.decoder(m, int(k), max_new, bool(ignore_eos))(ids)
        except Exception as e:
            rows.append([m, "error", "", "", "", "", "", f"{type(e).__name__}: {e}"])
            continue
        if ref is None:
            ref = toks
        d = first_divergence(ref, toks)
        acc = st.get("acceptance_rate")
        rows.append([m, st.get("n_tokens", len(toks)), f"{st.get('prefill_s', float('nan')) * 1e3:.1f}" if "prefill_s" in st else "-",
                     f"{st['latency_s']:.3f}", f"{st['tokens_per_s']:.1f}",
                     f"{st['peak_mem_mb']:.0f}" if st.get("peak_mem_mb") else "n/a (CPU)",
                     "yes" if d is None else f"NO (diverges at {d})",
                     f"acceptance {acc:.2f}, {st['mean_accepted_per_round']:.2f}/round" if acc is not None else ""])
    base = float(rows[1][3]) if rows[1][1] != "error" else None
    note = (f"Prompt: {ids.shape[1]} tokens; generating {max_new} tokens {'(EOS ignored)' if ignore_eos else '(stops at EOS)'}; "
            f"device {ENGINE.device}. Reference for token equality = uncached greedy. "
            + (f"KV-cache speed-up over uncached: {float(rows[0][3]) / base:.2f}x." if base else ""))
    return rows, note


def run_attention(seq_len, block, causal):
    dev = ENGINE.device if ENGINE else ("cuda" if torch.cuda.is_available() else "cpu")
    L, blk = int(seq_len), int(block)
    dtp = torch.float16 if dev.startswith("cuda") else torch.float32
    g = torch.Generator(device="cpu").manual_seed(L)
    q, kk, v = (torch.randn(1, 12, L, 128, generator=g).to(dev, dtp) for _ in range(3))
    gold = reference_attention(q.double(), kk.double(), v.double(), bool(causal))
    rows = []
    for name, fn in (("reference (materialised scores)", lambda: reference_attention(q, kk, v, bool(causal))),
                     ("tiled online-softmax (ours)", lambda: tiled_attention(q, kk, v, bool(causal), block_q=blk, block_k=blk)),
                     ("torch SDPA (fused kernel)", lambda: torch_sdpa(q, kk, v, bool(causal)))):
        fn()
        if dev.startswith("cuda"):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        t = time.perf_counter()
        o = fn()
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1e3
        mem = f"{(torch.cuda.max_memory_allocated() - base) / 2**20:.1f}" if dev.startswith("cuda") else "n/a (CPU)"
        e = (o.double() - gold).abs()
        rows.append([name, f"{ms:.2f}", mem, f"{float(e.max()):.2e}", f"{float(e.mean()):.2e}"])
    return rows, f"B=1, H=12, D=128 (Qwen2.5-1.5B head shape), L={L}, dtype={str(dtp).split('.')[-1]}, device={dev}; errors vs fp64 reference."


# ============================================================================ UI
def build_ui():
    names = [f"{e['patient_id']} / {e['trial_id']}" for e in EXAMPLES]
    with gr.Blocks(title="TrialMatch-Qwen screening prototype") as demo:
        gr.Markdown("# TrialMatch-Qwen: criterion-level clinical-trial eligibility screening")
        gr.Markdown(DISCLAIMER)
        payload = gr.State(None)
        with gr.Row():
            with gr.Column(scale=5):
                ex_dd = gr.Dropdown(names, label="Load an example (fictional demo or TRAINING-split patients only)", value=None)
                note = gr.Textbox(label="Patient note (free text or '0. sentence' numbered lines)", lines=10)
            with gr.Column(scale=5):
                with gr.Row():
                    nct = gr.Textbox(label="Fetch criteria from ClinicalTrials.gov (optional)", placeholder="NCT01397994", scale=3)
                    fetch_btn = gr.Button("Fetch", scale=1)
                fetch_msg = gr.Markdown()
                inc = gr.Textbox(label="Inclusion criteria (one per line)", lines=5)
                exc = gr.Textbox(label="Exclusion criteria (one per line)", lines=5)
        with gr.Row():
            mode = gr.Dropdown(MODES, value=MODES[0], label="Inference mode")
            k = gr.Slider(1, 8, value=4, step=1, label="Speculative draft length k")
            run_btn = gr.Button("Screen patient", variant="primary")
        with gr.Tab("Results"):
            summary = gr.Markdown()
            table = gr.Dataframe(headers=["#", "type", "criterion", "label", "sentence IDs", "evidence text",
                                          "explanation", "flags", "latency (s)"], wrap=True, interactive=False)
            with gr.Row():
                with gr.Column():
                    pick = gr.Dropdown([], label="Inspect a criterion")
                    detail = gr.Markdown()
                with gr.Column():
                    note_view = gr.HTML(label="Numbered note (cited evidence highlighted)")
            files = gr.File(label="Download screening report (HTML + JSON with provenance)", file_count="multiple")
        with gr.Tab("Inference benchmark (live)"):
            gr.Markdown("Runs every decoding implementation on the SAME prompt (criterion # below) and checks "
                        "greedy token equality against the uncached reference.")
            with gr.Row():
                b_idx = gr.Number(value=1, precision=0, label="Criterion #")
                b_new = gr.Slider(8, 256, value=64, step=8, label="max new tokens")
                b_ign = gr.Checkbox(value=True, label="ignore EOS (fixed-length timing)")
                b_btn = gr.Button("Run decoding benchmark", variant="primary")
            b_note = gr.Markdown()
            b_table = gr.Dataframe(headers=["mode", "tokens", "prefill (ms)", "latency (s)", "tokens/s", "peak mem (MB)",
                                            "identical to reference", "speculative stats"], interactive=False)
            gr.Markdown("### Attention kernels: reference vs tiled online-softmax vs fused SDPA")
            with gr.Row():
                a_len = gr.Slider(16, 4096, value=1024, step=16, label="sequence length")
                a_blk = gr.Slider(16, 256, value=64, step=16, label="tile size")
                a_causal = gr.Checkbox(value=True, label="causal")
                a_btn = gr.Button("Run attention check")
            a_note = gr.Markdown()
            a_table = gr.Dataframe(headers=["implementation", "time (ms)", "extra peak mem (MB)", "max |err|", "mean |err|"], interactive=False)
        with gr.Tab("About"):
            gr.Markdown(f"""**Model:** `{ENGINE.args.target if ENGINE else TARGET_MODEL}` + LoRA adapter `{ENGINE.args.adapter if ENGINE else None}`;
draft `{ENGINE.args.draft if ENGINE else DRAFT_MODEL}` (+ `{ENGINE.args.draft_adapter if ENGINE else None}`).
Labels follow the TrialGPT expert scheme (inclusion: included / not included / not enough information / not applicable;
exclusion: excluded / not excluded / not enough information / not applicable). Explanations are built only from the cited
note sentences. The overall outcome is an explicit rule-based aggregation, not a validated prediction.
App version {__version__}, commit {git_commit()}.""")

        ex_dd.change(load_example, ex_dd, [note, inc, exc])
        fetch_btn.click(fetch_trial, nct, [inc, exc, fetch_msg])
        run_btn.click(run_screen, [note, inc, exc, mode, k], [table, summary, note_view, files, payload, pick])
        pick.change(inspect, [pick, payload], [detail, note_view])
        b_btn.click(run_bench, [note, inc, exc, b_idx, k, b_new, b_ign], [b_table, b_note])
        a_btn.click(run_attention, [a_len, a_blk, a_causal], [a_table, a_note])
    return demo


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=os.environ.get("TM_TARGET", TARGET_MODEL))
    ap.add_argument("--adapter", default=os.environ.get("TM_ADAPTER", _default("adapters_export/target_qwen2.5-1.5b_lora")))
    ap.add_argument("--draft", default=os.environ.get("TM_DRAFT", DRAFT_MODEL))
    ap.add_argument("--draft_adapter", default=os.environ.get("TM_DRAFT_ADAPTER", _default("adapters_export/draft_qwen2.5-0.5b_lora")))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max_prompt_tokens", type=int, default=1536)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link (e.g. on Kaggle/Colab)")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    ENGINE = Engine(args)
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(server_name=args.host, server_port=args.port, share=args.share)


def _default(rel):
    p = ROOT / rel
    return str(p) if p.exists() else None


if __name__ == "__main__":
    main()
