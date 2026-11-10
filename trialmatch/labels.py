"""Label sets and robust parsing of model outputs.

The TrialGPT expert labels are preserved exactly (no binary collapse):
  inclusion: included | not included | not enough information | not applicable
  exclusion: excluded | not excluded | not enough information | not applicable
"""
from __future__ import annotations

import json
import re

INCLUSION_LABELS = ["included", "not included", "not enough information", "not applicable"]
EXCLUSION_LABELS = ["excluded", "not excluded", "not enough information", "not applicable"]
ALL_LABELS = ["included", "not included", "excluded", "not excluded",
              "not enough information", "not applicable"]
INVALID = "invalid"

LABEL_DEFINITIONS = {
    "inclusion": {
        "included": "the note shows the patient MEETS this inclusion criterion",
        "not included": "the note shows the patient does NOT meet this inclusion criterion",
        "not enough information": "the note does not contain enough information to decide",
        "not applicable": "the criterion is irrelevant to this patient",
    },
    "exclusion": {
        "excluded": "the note shows the patient MEETS this exclusion criterion (patient should be excluded)",
        "not excluded": "the note shows the patient does NOT meet this exclusion criterion",
        "not enough information": "the note does not contain enough information to decide",
        "not applicable": "the criterion is irrelevant to this patient",
    },
}


def normalize_type(ctype: str) -> str:
    c = str(ctype).strip().lower()
    if c.startswith("incl"):
        return "inclusion"
    if c.startswith("excl"):
        return "exclusion"
    raise ValueError(f"Unknown criterion type: {ctype!r}")


def allowed_labels(ctype: str) -> list[str]:
    return INCLUSION_LABELS if normalize_type(ctype) == "inclusion" else EXCLUSION_LABELS


# longest first so "not included" is matched before "included"
_LABEL_SEARCH_ORDER = sorted(ALL_LABELS, key=len, reverse=True)


def _find_label(text: str) -> str | None:
    t = text.lower()
    for lab in _LABEL_SEARCH_ORDER:
        if re.search(r'"label"\s*:\s*"' + re.escape(lab) + r'"', t):
            return lab
    # fallback: earliest occurrence of any label string ('not included' beats 'included')
    cands = []
    for lab in ALL_LABELS:
        for mm in re.finditer(r"(?<![a-z])" + re.escape(lab), t):
            i = mm.start()
            if lab in ("included", "excluded") and t[max(0, i - 4):i] == "not ":
                continue
            cands.append((i, -len(lab), lab))
    return min(cands)[2] if cands else None


def parse_output(text: str, ctype: str, valid_ids: list[int] | set[int] | None = None) -> dict:
    """Parse the model's JSON answer.  Never raises.

    Returns dict(label, evidence, json_ok, label_valid, dropped_ids, raw).
    A label that is not in the criterion type's label set is mapped to 'invalid'
    (scored as an error) -- we deliberately do not guess a mapping.
    """
    raw = text
    obj = None
    json_ok = False
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        cand = m.group(0)
        for attempt in (cand, cand[: cand.find("}") + 1]):
            try:
                obj = json.loads(attempt)
                json_ok = isinstance(obj, dict)
                break
            except Exception:
                continue
    label = None
    evidence: list[int] = []
    if json_ok:
        lab = obj.get("label")
        if isinstance(lab, str):
            label = lab.strip().lower()
        ev = obj.get("evidence", [])
        if isinstance(ev, (list, tuple)):
            for e in ev:
                try:
                    evidence.append(int(e))
                except Exception:
                    pass
    if label is None or label not in ALL_LABELS:
        label = _find_label(text)
    if not json_ok:
        em = re.search(r'"evidence"\s*:\s*\[([^\]]*)\]', text)
        if em:
            evidence = [int(x) for x in re.findall(r"-?\d+", em.group(1))]
    allowed = allowed_labels(ctype)
    label_valid = label in allowed
    if not label_valid:
        label = INVALID
    dropped = []
    if valid_ids is not None:
        vs = set(valid_ids)
        dropped = [e for e in evidence if e not in vs]
        evidence = [e for e in evidence if e in vs]
    evidence = sorted(set(evidence))
    return {"label": label, "evidence": evidence, "json_ok": json_ok,
            "label_valid": label_valid, "dropped_ids": dropped, "raw": raw}
