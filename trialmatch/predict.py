"""Screening pipeline: prompt -> decoder -> parsed label/evidence -> grounded explanation.

Explanations are built *extractively* from the cited note sentences, so every statement in
the explanation is traceable to the patient record (no free-text hallucination channel).
"""
from __future__ import annotations

import time

import torch

from .data import build_prompt_with_budget, parse_note, segment_free_text
from .labels import INVALID, normalize_type, parse_output


def explain(ctype: str, label: str, evidence: list[int], sent_map: dict[int, str], criterion: str) -> tuple[str, list[str]]:
    """Return (explanation, flags).  Absent evidence is never presented as proof."""
    flags = []
    quoted = " ".join(f'[{i}] "{sent_map[i]}"' for i in evidence if i in sent_map)
    meets = {"inclusion": "included", "exclusion": "excluded"}[ctype]
    fails = {"inclusion": "not included", "exclusion": "not excluded"}[ctype]
    if label == INVALID:
        flags.append("invalid-output")
        return "The model did not return a valid label for this criterion type; manual review required.", flags
    if label == meets:
        if evidence:
            exp = f"The patient appears to MEET this {ctype} criterion, based on {quoted}."
        else:
            exp = f"The model judged the patient to meet this {ctype} criterion but cited no sentence."
            flags.append("unsupported-positive")
    elif label == fails:
        if evidence:
            exp = f"The cited sentence(s) indicate the patient does NOT meet this {ctype} criterion: {quoted}."
        else:
            exp = (f"No sentence in the note documents this condition; the model judged it as not met by absence. "
                   f"Absence of documentation is not proof -- confirm before relying on it.")
            flags.append("absence-based")
    elif label == "not enough information":
        exp = "The note does not contain enough information to decide this criterion."
        if evidence:
            exp += f" Related context: {quoted}."
        exp += f" Missing information: {criterion.strip()}"
        flags.append("missing-information")
    else:  # not applicable
        exp = "The criterion was judged not applicable to this patient."
        if evidence:
            exp += f" Basis: {quoted}."
    return exp, flags


def result_record(ctype, criterion, parsed, sentences, extra=None) -> dict:
    ctype = normalize_type(ctype)
    sent_map = dict(sentences)
    exp, flags = explain(ctype, parsed["label"], parsed["evidence"], sent_map, criterion)
    rec = {"criterion_type": ctype, "criterion": criterion, "label": parsed["label"],
           "evidence_ids": parsed["evidence"],
           "evidence_text": [sent_map[i] for i in parsed["evidence"] if i in sent_map],
           "explanation": exp, "flags": flags, "json_ok": parsed["json_ok"],
           "dropped_ids": parsed["dropped_ids"], "raw_output": parsed["raw"]}
    if extra:
        rec.update(extra)
    return rec


@torch.no_grad()
def hf_generate_batch(model, tok, prompts: list[str], max_new_tokens: int, eos: list[int]) -> list[str]:
    """Library reference generation (pure greedy; Qwen's default repetition penalty disabled)."""
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                         temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                         pad_token_id=tok.pad_token_id, eos_token_id=eos)
    gen = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)


def prepare_note(note_text: str) -> list[tuple[int, str]]:
    return parse_note(note_text)


def screen_one(model, tok, sentences, ctype, criterion, decoder, max_prompt_tokens=1536, max_new_tokens=64):
    """decoder: callable(prompt_ids_tensor) -> (generated_token_ids, stats_dict)."""
    prompt, trunc = build_prompt_with_budget(tok, sentences, ctype, criterion, max_prompt_tokens)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    t0 = time.perf_counter()
    gen_ids, stats = decoder(ids)
    stats = dict(stats or {})
    stats.setdefault("latency_s", time.perf_counter() - t0)
    text = tok.decode(gen_ids, skip_special_tokens=True)
    kept = [i for i, _ in sentences if i in set(trunc["kept_sentence_ids"])]
    parsed = parse_output(text, ctype, kept)
    return result_record(ctype, criterion, parsed, sentences,
                         {"truncation": trunc, "stats": stats, "n_generated": len(gen_ids)})


__all__ = ["explain", "result_record", "hf_generate_batch", "screen_one", "segment_free_text", "prepare_note"]
