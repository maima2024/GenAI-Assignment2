import numpy as np
import pandas as pd
from trialmatch.data import make_split, leakage_checks, parse_note, segment_free_text, parse_id_list


def fake_df():
    rows = []
    for p in range(53):
        for c in range(3):
            rows.append({"annotation_id": p * 3 + c, "patient_id": f"sigir-2014{p}", "note": f"0. note {p}",
                         "trial_id": f"NCT{c}", "criterion_type": "inclusion", "criterion_text": "x",
                         "expert_eligibility": "included", "expert_sentence_ids": [0]})
    return pd.DataFrame(rows)


def test_split_sizes_and_leakage():
    df = fake_df()
    s = make_split(df)
    assert [len(s[k]) for k in ("train", "val", "test")] == [37, 8, 8]
    ids = np.array(sorted(df.patient_id.unique())); np.random.default_rng(42).shuffle(ids)
    assert s["train"] == ids[:37].tolist()
    assert leakage_checks(df, s)["all_passed"]


def test_note_parsing():
    assert parse_note("0. A b.\n1. C d.")[1] == (1, "C d.")
    assert len(segment_free_text("He is 45. Dr. Smith saw him. BP 120/80.")) == 3
    assert parse_id_list("[3, 1]") == [1, 3] and parse_id_list("[]") == []
