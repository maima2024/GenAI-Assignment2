from trialmatch.labels import parse_output, INVALID
from trialmatch.metrics import evidence_prf, evidence_metrics, classification_metrics
from trialmatch.labels import INCLUSION_LABELS


def test_parse_json():
    r = parse_output('{"evidence": [2, 0, 9], "label": "not included"}', "inclusion", [0, 1, 2])
    assert r["label"] == "not included" and r["evidence"] == [0, 2] and r["dropped_ids"] == [9] and r["json_ok"]


def test_parse_fallback_and_invalid():
    assert parse_output('label: not enough information', "exclusion")["label"] == "not enough information"
    assert parse_output('The patient is included.', "inclusion")["label"] == "included"
    assert parse_output('{"evidence": [], "label": "excluded"}', "inclusion")["label"] == INVALID
    assert parse_output('garbage', "inclusion")["label"] == INVALID


def test_evidence_conventions():
    assert evidence_prf(set(), set()) == (1.0, 1.0, 1.0)
    assert evidence_prf({1}, set()) == (0.0, 0.0, 0.0)
    assert evidence_prf(set(), {1}) == (0.0, 0.0, 0.0)
    p, r, f = evidence_prf({1, 2}, {2, 3})
    assert abs(p - .5) < 1e-9 and abs(r - .5) < 1e-9
    m = evidence_metrics([[1, 2], [], []], [[2], [], [4]])
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1 and m["empty_agreement_rate"] == 0.5


def test_classification():
    m = classification_metrics(["included", "not included", "included"], ["included", "invalid", "not included"], INCLUSION_LABELS)
    assert abs(m["accuracy"] - 1 / 3) < 1e-9 and "invalid" in m["confusion_matrix"]["labels"]
