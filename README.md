# TrialMatch-Qwen — criterion-level clinical-trial eligibility screening with a fine-tuned decoder Transformer

> **Educational research prototype — NOT a clinical enrollment decision tool.**

Given (i) a patient note, (ii) a criterion type (inclusion/exclusion) and (iii) the criterion text, a LoRA-fine-tuned
**Qwen2.5-1.5B-Instruct** returns a structured answer `{"evidence": [sentence IDs], "label": "..."}` using the full
TrialGPT expert label set (inclusion: *included / not included / not enough information / not applicable*; exclusion:
*excluded / not excluded / not enough information / not applicable*). Explanations are built extractively from the cited
sentences. The repository also contains hand-written implementations of **KV-cached decoding**, **tiled online-softmax
attention** and **greedy speculative decoding** (Qwen2.5-0.5B draft), with correctness tests and benchmarks, and a
**browser application** (Gradio).

| Task | Code |
|---|---|
| Fixed 37/8/8 patient split (seed 42), manifest, leakage checks | `trialmatch/data.py`, `trialmatch/split.py` → `results/split/` |
| Prompt / output schema / truncation logging | `trialmatch/data.py`, `trialmatch/labels.py` |
| LoRA fine-tuning (answer-only loss) + 0.5B hyper-parameter sweep | `trialmatch/train.py`, `scripts/run_all.sh` |
| Before/after evaluation, metrics, error analysis | `trialmatch/evaluate.py`, `trialmatch/metrics.py`, `scripts/error_analysis.py` |
| KV cache: uncached reference, explicit-cache HF loop, from-scratch decoder with own cache | `trialmatch/kv_decode.py`, `benchmarks/bench_kv.py` |
| Reference vs tiled online-softmax attention | `trialmatch/attention.py`, `benchmarks/bench_attention.py`, `tests/` |
| Greedy speculative decoding (draft → verify → accept → correct → cache rollback) | `trialmatch/speculative.py`, `benchmarks/bench_speculative.py` |
| Browser app (screening, evidence, report export, live benchmarks) | `app/app.py` |
| Rule-based preliminary patient–trial outcome | `trialmatch/aggregate.py` |

## 1. Reproduce everything (free Kaggle/Colab T4, ~60–90 min)
Open `notebooks/kaggle_run_all.ipynb` on Kaggle (GPU T4, Internet on) and run all cells, or:
```bash
pip install -r requirements.txt
bash scripts/run_all.sh            # split → sweep → fine-tune → eval → attention → KV → speculative → figures
python -m pytest -q tests
```
Outputs: `results/split/split_manifest.json` (patient IDs + counts), `results/eval/*/metrics.json|predictions.jsonl`,
`results/bench/*.json` (benchmark logs), `results/figures/`, `results/summary.json`, `results/logs/`, adapters in `adapters_export/`.

## 2. Model weights
LoRA adapters (a few tens of MB) are hosted on the Hugging Face Hub (base weights are downloaded automatically from Qwen):
* target: `https://huggingface.co/<HF_USER>/trialmatch-qwen2.5-1.5b-lora`
* draft:  `https://huggingface.co/<HF_USER>/trialmatch-qwen2.5-0.5b-draft-lora`

Upload with `HF_TOKEN=... python scripts/push_to_hub.py --user <HF_USER>`.

## 3. Start the browser application
```bash
pip install -r requirements.txt
# with local adapters (after run_all.sh) – picked up automatically:
python -m app.app
# or with the published adapters:
python -m app.app --adapter <HF_USER>/trialmatch-qwen2.5-1.5b-lora --draft_adapter <HF_USER>/trialmatch-qwen2.5-0.5b-draft-lora
```
Open **http://127.0.0.1:7860**. On Kaggle/Colab add `--share` to get a public `https://*.gradio.live` link.
The app runs on a GPU or (slower, ~3–8 s per criterion) on CPU. No notebook or IDE is needed to use it.

Workflow: paste a patient note (free text is auto-split and numbered; `0. ...` numbered notes keep their IDs) → type
inclusion/exclusion criteria (one per line) or fetch them by NCT ID from ClinicalTrials.gov → choose an inference mode
(KV-cached, speculative, uncached reference, scratch decoder with tiled attention, HF generate) → **Screen patient** →
inspect each criterion (cited sentences are highlighted) → download the HTML/JSON report (with provenance).
The *Inference benchmark* tab runs all decoders live on the same prompt and checks token equality; it also compares the
attention kernels. Invalid inputs (empty note, no criteria, too many/too long criteria, bad NCT ID, network failure)
produce an error message and the app keeps running.

## 4. Design summary
* **Split**: `ids = sorted(unique patient_id)`, `numpy.random.default_rng(42).shuffle(ids)`, 37/8/8. All annotations of a
  patient live in one split; the test split is used only for the final evaluation.
* **Inputs** are only `note`, `criterion_type`, `criterion_text`. The `gpt4_*` columns and the dataset's `training` flag are never used.
* **Truncation**: prompt budget 1,536 tokens; if exceeded, trailing note sentences are dropped and their IDs logged
  (and whether gold evidence was lost). Notes in this dataset fit, so no truncation occurs in practice (see logs).
* **Decoding for evaluation**: greedy, `max_new_tokens=64`, `repetition_penalty=1.0`, fp16, batched with left padding.

## 5. Limitations
Small single-site dataset (53 patients); criterion-level labels only; the patient–trial outcome in the app is a
transparent rule, not a validated prediction. See the report for details.

## AI-use statement
AI assistants were used to help write code and draft text, as permitted by the assignment; all results come from the
scripts in this repository.
