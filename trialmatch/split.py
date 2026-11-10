"""Create and publish the fixed 37/8/8 patient-grouped split.

    python -m trialmatch.split --out results/split
"""
import argparse
import json
from pathlib import Path

from .data import build_manifest, load_raw, make_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/split")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    print("columns:", list(df.columns))
    print("rows:", len(df), "patients:", df.patient_id.nunique())
    split = make_split(df)
    man = build_manifest(df, split)
    (out / "split_manifest.json").write_text(json.dumps(man, indent=2))

    rows = []
    for name, pids in split.items():
        for _, r in df[df.patient_id.isin(pids)].iterrows():
            rows.append(f"{int(r.annotation_id)},{r.patient_id},{r.trial_id},{r.criterion_type},{name}")
    (out / "split_assignments.csv").write_text(
        "annotation_id,patient_id,trial_id,criterion_type,split\n" + "\n".join(sorted(rows, key=lambda s: int(s.split(',')[0]))))

    # a few TRAINING-split cases as demo examples for the app (never val/test)
    tr = df[df.patient_id.isin(split["train"])]
    examples = []
    for pid in split["train"][:5]:
        sub = tr[tr.patient_id == pid]
        t0 = sub.trial_id.iloc[0]
        st = sub[sub.trial_id == t0]
        examples.append({"patient_id": pid, "trial_id": t0, "trial_title": st.trial_title.iloc[0],
                         "note": st.note.iloc[0],
                         "inclusion": st[st.criterion_type == "inclusion"].criterion_text.tolist(),
                         "exclusion": st[st.criterion_type == "exclusion"].criterion_text.tolist()})
    (out / "app_examples.json").write_text(json.dumps(examples, indent=2))

    print(json.dumps({k: {kk: v[kk] for kk in ("n_patients", "n_annotations", "label_counts")}
                      for k, v in man["splits"].items()}, indent=2))
    print("leakage checks:", man["leakage"])
    if not man["leakage"]["all_passed"]:
        raise SystemExit("LEAKAGE CHECK FAILED")


if __name__ == "__main__":
    main()
