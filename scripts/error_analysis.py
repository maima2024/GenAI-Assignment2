"""Categorise test-set errors of the fine-tuned model (and contrast with zero-shot)."""
import json
from collections import Counter
from pathlib import Path

R = Path("results")
OUT = R / "error_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def load_preds(tag):
    p = R / "eval" / tag / "predictions.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def categories(r):
    g, p = r["gold_label"], r["pred_label"]
    ge, pe = set(r["gold_evidence"]), set(r["pred_evidence"])
    cats = []
    if p == "invalid":
        cats.append("invalid/unparseable output")
    if g == "not enough information" and p != g:
        cats.append("missed missing-information (over-confident)")
    if p == "not enough information" and g != p:
        cats.append("false missing-information (evidence ignored)")
    if "not applicable" in (g, p) and g != p:
        cats.append("not-applicable confusion")
    pol = {("included", "not included"), ("not included", "included"), ("excluded", "not excluded"), ("not excluded", "excluded")}
    if (g, p) in pol:
        cats.append("polarity flip")
    if pe and not (pe & ge):
        cats.append("unsupported evidence (no overlap with expert)")
    if ge and not pe:
        cats.append("missing evidence (expert cited, model none)")
    if p in ("not included", "not excluded") and not pe and g != p:
        cats.append("absence treated as proof")
    if p in ("included", "excluded") and not pe:
        cats.append("positive decision without evidence")
    return cats or ["other label confusion"]


ft = load_preds("ft_test")
base = {r["annotation_id"]: r for r in load_preds("base_test")}
errs = []
for r in ft:
    label_err = r["gold_label"] != r["pred_label"]
    ev_err = set(r["gold_evidence"]) != set(r["pred_evidence"])
    if not (label_err or ev_err):
        continue
    sm = dict((int(i), s) for i, s in r.get("sentences", [])) if "sentences" in r else {}
    errs.append({"annotation_id": r["annotation_id"], "patient_id": r["patient_id"], "trial_id": r["trial_id"],
                 "criterion_type": r["criterion_type"], "criterion": r["criterion"],
                 "gold_label": r["gold_label"], "pred_label": r["pred_label"],
                 "gold_evidence": r["gold_evidence"], "pred_evidence": r["pred_evidence"],
                 "pred_evidence_text": r["evidence_text"], "explanation": r["explanation"], "flags": r["flags"],
                 "label_error": label_err, "evidence_only_error": (not label_err) and ev_err,
                 "categories": categories(r) if label_err else (["evidence mismatch only"] + [c for c in categories(r) if "evidence" in c]),
                 "zero_shot_label": base.get(r["annotation_id"], {}).get("pred_label"),
                 "raw_output": r["raw_output"]})
with open(OUT / "test_errors.jsonl", "w") as f:
    for e in errs:
        f.write(json.dumps(e) + "\n")
cnt = Counter(c for e in errs for c in e["categories"])
conf = Counter((e["gold_label"], e["pred_label"]) for e in errs if e["label_error"])
# pick a diverse set of >= 10 label errors covering as many categories as possible
picked, seen = [], set()
for e in sorted([e for e in errs if e["label_error"]], key=lambda e: -len(set(e["categories"]) - seen)):
    if set(e["categories"]) - seen or len(picked) < 10:
        picked.append(e)
        seen |= set(e["categories"])
    if len(picked) >= 12:
        break
summary = {"n_test": len(ft), "n_with_any_error": len(errs), "n_label_errors": sum(e["label_error"] for e in errs),
           "n_evidence_only_errors": sum(e["evidence_only_error"] for e in errs),
           "category_counts": dict(cnt.most_common()),
           "label_confusions": {f"{g} -> {p}": n for (g, p), n in conf.most_common()},
           "selected_examples": picked}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "selected_examples"}, indent=2))
