"""Classification and evidence-retrieval metrics.

Evidence conventions (stated in the report):
  * per-record (macro) P/R/F1:  gold = {} and pred = {}  -> P = R = F1 = 1  (correct abstention)
                                exactly one of them empty -> P = R = F1 = 0
  * micro P/R/F1: TP/FP/FN summed over all records (empty/empty records contribute nothing)
  * macro over the subset with non-empty gold evidence is also reported (standard IR view).
Predicted IDs that do not exist in the note are removed before scoring and counted separately.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .labels import ALL_LABELS, EXCLUSION_LABELS, INCLUSION_LABELS, INVALID


def classification_metrics(gold: list[str], pred: list[str], label_set: list[str]) -> dict:
    gold = list(gold)
    pred = list(pred)
    n = len(gold)
    if n == 0:
        return {"n": 0}
    present = [l for l in label_set if l in set(gold)]  # macro over classes present in gold
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=label_set, zero_division=0)
    per_class = {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                 for i, lab in enumerate(label_set)}
    macro_f1 = float(np.mean([per_class[l]["f1"] for l in present])) if present else 0.0
    cm_labels = label_set + ([INVALID] if INVALID in pred else [])
    cm = confusion_matrix(gold, pred, labels=cm_labels)
    return {
        "n": n,
        "accuracy": float(np.mean([g == q for g, q in zip(gold, pred)])),
        "macro_f1": macro_f1,
        "macro_f1_classes": present,
        "per_class": per_class,
        "invalid_rate": float(np.mean([q == INVALID for q in pred])),
        "confusion_matrix": {"labels": cm_labels, "matrix": cm.tolist(),
                             "rows": "gold", "cols": "predicted"},
    }


def evidence_prf(gold: set, pred: set) -> tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not gold or not pred:
        return 0.0, 0.0, 0.0
    tp = len(gold & pred)
    p = tp / len(pred)
    r = tp / len(gold)
    f = 0.0 if tp == 0 else 2 * p * r / (p + r)
    return p, r, f


def evidence_metrics(gold_sets: list, pred_sets: list) -> dict:
    gold_sets = [set(g) for g in gold_sets]
    pred_sets = [set(p) for p in pred_sets]
    per = [evidence_prf(g, p) for g, p in zip(gold_sets, pred_sets)]
    tp = sum(len(g & p) for g, p in zip(gold_sets, pred_sets))
    fp = sum(len(p - g) for g, p in zip(gold_sets, pred_sets))
    fn = sum(len(g - p) for g, p in zip(gold_sets, pred_sets))
    mp = tp / (tp + fp) if tp + fp else 0.0
    mr = tp / (tp + fn) if tp + fn else 0.0
    mf = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
    nonempty = [i for i, g in enumerate(gold_sets) if g]
    per_ne = [per[i] for i in nonempty]
    both_empty = sum(1 for g, p in zip(gold_sets, pred_sets) if not g and not p)
    gold_empty = sum(1 for g in gold_sets if not g)
    return {
        "n": len(gold_sets),
        "macro_precision": float(np.mean([x[0] for x in per])) if per else 0.0,
        "macro_recall": float(np.mean([x[1] for x in per])) if per else 0.0,
        "macro_f1": float(np.mean([x[2] for x in per])) if per else 0.0,
        "micro_precision": mp, "micro_recall": mr, "micro_f1": mf,
        "nonempty_gold_n": len(nonempty),
        "nonempty_gold_macro_precision": float(np.mean([x[0] for x in per_ne])) if per_ne else 0.0,
        "nonempty_gold_macro_recall": float(np.mean([x[1] for x in per_ne])) if per_ne else 0.0,
        "nonempty_gold_macro_f1": float(np.mean([x[2] for x in per_ne])) if per_ne else 0.0,
        "gold_empty_n": gold_empty,
        "empty_agreement_rate": both_empty / gold_empty if gold_empty else None,
        "tp": tp, "fp": fp, "fn": fn,
    }


def full_report(records: list[dict]) -> dict:
    """records: dicts with gold_label, pred_label, criterion_type, gold_evidence, pred_evidence."""
    out = {}
    groups = {"overall": records,
              "inclusion": [r for r in records if r["criterion_type"] == "inclusion"],
              "exclusion": [r for r in records if r["criterion_type"] == "exclusion"]}
    sets = {"overall": ALL_LABELS, "inclusion": INCLUSION_LABELS, "exclusion": EXCLUSION_LABELS}
    for g, recs in groups.items():
        out[g] = {
            "classification": classification_metrics([r["gold_label"] for r in recs],
                                                     [r["pred_label"] for r in recs], sets[g]),
            "evidence": evidence_metrics([r["gold_evidence"] for r in recs],
                                         [r["pred_evidence"] for r in recs]),
        }
    out["json_parse_rate"] = float(np.mean([r.get("json_ok", True) for r in records])) if records else 0.0
    out["dropped_invalid_ids"] = int(sum(len(r.get("dropped_ids", [])) for r in records))
    return out
