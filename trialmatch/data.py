"""Dataset loading, the fixed patient-grouped split, prompt construction and truncation.

Only three fields of a row are ever used as model input: the patient note, the criterion
type and the criterion text.  The previous model outputs (gpt4_*) and the dataset's own
`training` flag are never used for input, targets, selection or evaluation.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import LABEL_DEFINITIONS, allowed_labels, normalize_type

DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"
SEED = 42
N_TRAIN, N_VAL, N_TEST = 37, 8, 8
FORBIDDEN_COLUMNS = ["gpt4_explanation", "explanation_correctness", "gpt4_sentences",
                     "gpt4_eligibility", "training"]
INPUT_COLUMNS = ["note", "criterion_type", "criterion_text"]
TARGET_COLUMNS = ["expert_eligibility", "expert_sentences"]

SYSTEM_PROMPT = (
    "You are a clinical-trial eligibility screening assistant used for research. "
    "You judge ONE eligibility criterion at a time using ONLY the numbered patient note. "
    "Do not assume facts that are not written in the note. "
    "Reply with a single JSON object and nothing else."
)


# ----------------------------------------------------------------------------- loading
def load_raw(cache_dir: str | None = None) -> pd.DataFrame:
    """Load the mandatory TrialGPT criterion annotations (single 'train' split, 1,015 rows)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="train", cache_dir=cache_dir)
    df = ds.to_pandas()
    df["patient_id"] = df["patient_id"].astype(str)
    df["criterion_type"] = df["criterion_type"].map(normalize_type)
    df["expert_eligibility"] = df["expert_eligibility"].astype(str).str.strip().str.lower()
    df["expert_sentence_ids"] = df["expert_sentences"].map(parse_id_list)
    return df


def parse_id_list(x) -> list[int]:
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return sorted({int(v) for v in x})
    s = str(x).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return sorted({int(i) for i in v})
        return [int(v)]
    except Exception:
        return sorted({int(i) for i in re.findall(r"\d+", s)})


# ------------------------------------------------------------------------ note handling
_NUM_LINE = re.compile(r"^\s*(\d+)\.\s*(.*)$")


def parse_note(note: str) -> list[tuple[int, str]]:
    """Return [(sentence_id, text)].  Uses the dataset's own '<id>. sentence' numbering
    when present (original IDs preserved), otherwise segments free text."""
    lines = [ln for ln in str(note).splitlines() if ln.strip()]
    parsed = []
    for ln in lines:
        m = _NUM_LINE.match(ln)
        if not m:
            parsed = None
            break
        parsed.append((int(m.group(1)), m.group(2).strip()))
    if parsed:
        return parsed
    return segment_free_text(note)


def segment_free_text(text: str) -> list[tuple[int, str]]:
    """Sentence-split an un-numbered clinical note (used by the app for unseen notes)."""
    text = re.sub(r"[ \t]+", " ", str(text)).strip()
    chunks = []
    for para in re.split(r"\n+", text):
        para = para.strip()
        if not para:
            continue
        # split after . ! ? when followed by whitespace + capital/digit; keep decimals / abbreviations like "Dr."
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", para)
        merged: list[str] = []
        for p in parts:
            if merged and re.search(r"\b(?:Dr|Mr|Mrs|Ms|vs|e\.g|i\.e|approx|St)\.$", merged[-1]):
                merged[-1] += " " + p
            else:
                merged.append(p)
        chunks.extend(c.strip() for c in merged if c.strip())
    return list(enumerate(chunks))


def format_note(sentences: list[tuple[int, str]]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in sentences)


# ---------------------------------------------------------------------------- prompts
def user_message(note_sentences, ctype: str, ctext: str) -> str:
    ctype = normalize_type(ctype)
    labs = allowed_labels(ctype)
    defs = "\n".join(f'- "{lab}": {LABEL_DEFINITIONS[ctype][lab]}' for lab in labs)
    return (
        "Patient note (each sentence starts with its ID):\n"
        f"{format_note(note_sentences)}\n\n"
        f"Criterion type: {ctype}\n"
        f"Criterion: {str(ctext).strip()}\n\n"
        f"Allowed labels for an {ctype} criterion:\n{defs}\n\n"
        "First list the IDs of the note sentences that are relevant evidence for this criterion "
        "(empty list if none), then give the label.\n"
        'Answer format: {"evidence": [IDs], "label": "<allowed label>"}'
    )


def target_text(evidence_ids: list[int], label: str) -> str:
    return json.dumps({"evidence": sorted(int(i) for i in evidence_ids), "label": label})


def chat_prompt(tokenizer, note_sentences, ctype, ctext) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(note_sentences, ctype, ctext)}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_prompt_with_budget(tokenizer, note_sentences, ctype, ctext, max_prompt_tokens: int):
    """Build the chat prompt; if it exceeds the token budget, drop trailing note sentences
    one at a time (never the criterion) and record exactly which sentence IDs were removed."""
    sents = list(note_sentences)
    removed: list[int] = []
    prompt = chat_prompt(tokenizer, sents, ctype, ctext)
    n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    original_tokens = n_tok
    while n_tok > max_prompt_tokens and len(sents) > 1:
        removed.insert(0, sents.pop()[0])
        prompt = chat_prompt(tokenizer, sents, ctype, ctext)
        n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    return prompt, {"prompt_tokens": n_tok, "original_prompt_tokens": original_tokens,
                    "truncated": bool(removed), "removed_sentence_ids": removed,
                    "kept_sentence_ids": [i for i, _ in sents]}


def row_to_example(row, tokenizer, max_prompt_tokens: int) -> dict:
    sents = parse_note(row["note"])
    prompt, trunc = build_prompt_with_budget(tokenizer, sents, row["criterion_type"],
                                             row["criterion_text"], max_prompt_tokens)
    gold_ids = list(row["expert_sentence_ids"])
    trunc["gold_evidence_lost"] = sorted(set(gold_ids) & set(trunc["removed_sentence_ids"]))
    return {
        "annotation_id": int(row["annotation_id"]),
        "patient_id": row["patient_id"],
        "trial_id": row.get("trial_id", ""),
        "criterion_type": row["criterion_type"],
        "criterion_text": row["criterion_text"],
        "sentences": sents,
        "prompt": prompt,
        "target": target_text(gold_ids, row["expert_eligibility"]),
        "gold_label": row["expert_eligibility"],
        "gold_evidence": gold_ids,
        "truncation": trunc,
    }


# ------------------------------------------------------------------------------ split
def make_split(df: pd.DataFrame) -> dict[str, list[str]]:
    """Sort unique patient IDs, shuffle with numpy default_rng(42), take 37/8/8."""
    ids = np.array(sorted(df["patient_id"].astype(str).unique()))
    if len(ids) != N_TRAIN + N_VAL + N_TEST:
        raise ValueError(f"Expected 53 unique patients, found {len(ids)}")
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    ids = ids.tolist()
    return {"train": ids[:N_TRAIN], "val": ids[N_TRAIN:N_TRAIN + N_VAL],
            "test": ids[N_TRAIN + N_VAL:]}


def _sha(items) -> str:
    return hashlib.sha256("\n".join(map(str, items)).encode()).hexdigest()


def leakage_checks(df: pd.DataFrame, split: dict[str, list[str]]) -> dict:
    s = {k: set(v) for k, v in split.items()}
    checks = {
        "patients_disjoint": not (s["train"] & s["val"] or s["train"] & s["test"] or s["val"] & s["test"]),
        "all_patients_assigned": set().union(*s.values()) == set(df["patient_id"]),
    }
    sub = {k: df[df.patient_id.isin(v)] for k, v in s.items()}
    ann = {k: set(v["annotation_id"]) for k, v in sub.items()}
    checks["annotations_disjoint"] = not (ann["train"] & ann["val"] or ann["train"] & ann["test"] or ann["val"] & ann["test"])
    checks["all_annotations_assigned"] = sum(len(a) for a in ann.values()) == len(df)
    notes = {k: set(v["note"].map(lambda x: hashlib.md5(x.encode()).hexdigest())) for k, v in sub.items()}
    checks["patient_notes_disjoint"] = not (notes["train"] & notes["val"] or notes["train"] & notes["test"] or notes["val"] & notes["test"])
    checks["forbidden_columns_unused_in_inputs"] = all(c not in INPUT_COLUMNS for c in FORBIDDEN_COLUMNS)
    trials = {k: set(v["trial_id"]) for k, v in sub.items()}
    info = {"trial_overlap_train_test": len(trials["train"] & trials["test"]),
            "trial_overlap_train_val": len(trials["train"] & trials["val"]),
            "note": "Trials may legitimately recur across splits (different patients); patients never do."}
    return {"checks": checks, "all_passed": all(checks.values()), "info": info}


def build_manifest(df: pd.DataFrame, split: dict[str, list[str]]) -> dict:
    out = {"dataset": DATASET_ID, "n_rows": int(len(df)), "seed": SEED,
           "procedure": "ids = sorted(unique(patient_id)); rng = numpy.random.default_rng(42); "
                        "rng.shuffle(ids); train = ids[:37]; val = ids[37:45]; test = ids[45:53]",
           "splits": {}}
    for name, pids in split.items():
        sub = df[df.patient_id.isin(pids)]
        out["splits"][name] = {
            "patient_ids": pids,
            "n_patients": len(pids),
            "n_annotations": int(len(sub)),
            "n_inclusion": int((sub.criterion_type == "inclusion").sum()),
            "n_exclusion": int((sub.criterion_type == "exclusion").sum()),
            "label_counts": {k: int(v) for k, v in sub.expert_eligibility.value_counts().items()},
            "n_empty_gold_evidence": int((sub.expert_sentence_ids.map(len) == 0).sum()),
            "per_patient_counts": {k: int(v) for k, v in sub.patient_id.value_counts().sort_index().items()},
            "annotation_ids_sha256": _sha(sorted(sub.annotation_id.tolist())),
        }
    out["leakage"] = leakage_checks(df, split)
    return out


def load_split_frames(manifest_path: str | Path, cache_dir=None) -> dict[str, pd.DataFrame]:
    man = json.loads(Path(manifest_path).read_text())
    df = load_raw(cache_dir)
    frames = {}
    for name, info in man["splits"].items():
        sub = df[df.patient_id.isin(info["patient_ids"])].sort_values("annotation_id").reset_index(drop=True)
        assert len(sub) == info["n_annotations"], f"{name}: count mismatch vs manifest"
        frames[name] = sub
    return frames
