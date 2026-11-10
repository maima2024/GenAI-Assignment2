"""Pick the fine-tuning configuration with the best VALIDATION macro-F1 (test never used).

    python scripts/select_config.py results/eval/sweep_*_val --out results/selection.json
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--out", default="results/selection.json")
a = ap.parse_args()
rows = []
for d in a.dirs:
    m = json.loads((Path(d) / "metrics.json").read_text())
    o = m["overall"]
    rows.append({"run": Path(d).name, "adapter": m["meta"]["adapter"],
                 "val_accuracy": o["classification"]["accuracy"], "val_macro_f1": o["classification"]["macro_f1"],
                 "val_evidence_micro_f1": o["evidence"]["micro_f1"]})
rows.sort(key=lambda r: (r["val_macro_f1"], r["val_accuracy"]), reverse=True)
best = rows[0]
Path(a.out).write_text(json.dumps({"criterion": "validation macro-F1 (tie: accuracy)", "candidates": rows, "best": best}, indent=2))
print(json.dumps(rows, indent=2))
print("BEST:", best["run"])
print(best["run"].split("sweep_")[-1].replace("_val", ""))
