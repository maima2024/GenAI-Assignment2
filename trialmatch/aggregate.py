"""Explicit rule-based aggregation of criterion-level predictions into a PRELIMINARY
patient-trial screening outcome.  The dataset has no trial-level ground truth; this verdict
is a transparent rule, not a learned or validated prediction.

Rules (applied in order):
  R1  any inclusion criterion 'not included'  OR any exclusion criterion 'excluded'
        -> "Likely ineligible (preliminary)"
  R2  else any 'not enough information' or 'invalid' prediction
        -> "Undetermined - more information needed"
  R3  else -> "Potentially eligible (preliminary)"
Absence-based decisions (a 'not excluded'/'not included' with no cited sentence) are listed
separately for human verification and never upgrade the outcome on their own.
"""
from __future__ import annotations


def aggregate(results: list[dict]) -> dict:
    fails = [r for r in results if (r["criterion_type"] == "inclusion" and r["label"] == "not included")
             or (r["criterion_type"] == "exclusion" and r["label"] == "excluded")]
    unknown = [r for r in results if r["label"] in ("not enough information", "invalid")]
    absence = [r for r in results if "absence-based" in r.get("flags", [])]
    if fails:
        verdict, rule = "Likely ineligible (preliminary)", "R1"
    elif unknown:
        verdict, rule = "Undetermined - more information needed", "R2"
    else:
        verdict, rule = "Potentially eligible (preliminary)", "R3"
    counts = {}
    for r in results:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    return {"verdict": verdict, "rule": rule, "n_criteria": len(results), "label_counts": counts,
            "failing_criteria": [r["criterion"] for r in fails],
            "missing_information": [r["criterion"] for r in unknown],
            "absence_based_for_review": [r["criterion"] for r in absence]}
