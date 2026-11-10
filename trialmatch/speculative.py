"""Greedy speculative decoding (draft -> verify -> accept prefix -> correct), implemented by hand.

Invariant kept for BOTH models: the KV cache holds exactly seq[:cache_len]; every forward pass
feeds seq[cache_len:] (+ new draft tokens) at absolute positions [cache_len, ...).

One round (seq ends with a 'pending' token whose KV is not cached yet in the target):
  1. Draft: k greedy steps of the draft model -> d_1..d_k.
  2. Verify: ONE target forward over [pending, d_1..d_k] (k+1 positions, causal) ->
     target greedy predictions t_0..t_k, where t_i follows seq + d_1..d_i.
  3. Accept the longest prefix with d_i == t_{i-1}; call its length n (0..k).
  4. Append d_1..d_n and then t_n (the target's correction if n < k, or a free 'bonus' token if n = k).
  5. Roll back caches: target cache is cropped to len(seq)-1 (drops KV of rejected drafts);
     draft cache is cropped likewise if it holds rejected tokens.
Because every emitted token equals the target's own greedy argmax given the same prefix, the output
is identical to ordinary greedy decoding of the target (up to floating-point ties).
"""
from __future__ import annotations

import time

import torch

from .kv_decode import DecodeResult, _peak_mem, _start_mem, _sync, model_forward, new_dynamic_cache


def check_compatible(tok_target, tok_draft, target, draft) -> dict:
    same_vocab = tok_target.get_vocab() == tok_draft.get_vocab()
    return {"same_tokenizer_vocab": same_vocab,
            "target_vocab_size": target.config.vocab_size, "draft_vocab_size": draft.config.vocab_size,
            "embedding_rows_equal": target.config.vocab_size == draft.config.vocab_size}


def _step(model, cache, new_ids: list[int], past_len: int, device, all_logits: bool):
    T = len(new_ids)
    pos = torch.arange(past_len, past_len + T, device=device)
    out = model_forward(model, 0 if all_logits else 1,
                        input_ids=torch.tensor([new_ids], device=device),
                        attention_mask=torch.ones(1, past_len + T, dtype=torch.long, device=device),
                        position_ids=pos[None], cache_position=pos, past_key_values=cache, use_cache=True)
    return out.logits[0], out.past_key_values


@torch.no_grad()
def speculative_greedy(target, draft, input_ids: torch.Tensor, max_new_tokens: int, k: int = 4,
                       eos: list[int] | None = None, ignore_eos: bool = False) -> DecodeResult:
    dev = input_ids.device
    _start_mem(dev)
    eos = set(eos or [])
    prompt = input_ids[0].tolist()
    t_cache, d_cache = new_dynamic_cache(target), new_dynamic_cache(draft)
    _sync(dev)
    t0 = time.perf_counter()

    # prefill target -> first token (exactly as plain greedy)
    lg, t_cache = _step(target, t_cache, prompt, 0, dev, all_logits=False)
    first = int(lg[-1].argmax())
    seq = prompt + [first]
    t_len = len(prompt)
    # prefill draft on the prompt
    _, d_cache = _step(draft, d_cache, prompt, 0, dev, all_logits=False)
    d_len = len(prompt)
    _sync(dev)
    t_first = time.perf_counter()

    gen = [first]
    rounds, drafted, accepted = 0, 0, 0
    accept_hist = [0] * (k + 1)
    draft_time = verify_time = 0.0
    done = (first in eos and not ignore_eos) or max_new_tokens <= 1
    while not done:
        rounds += 1
        # ---------------- 1) draft k tokens greedily (draft cache catches up on any un-cached tokens)
        ta = time.perf_counter()
        drafts = []
        d_in = seq[d_len:]
        for _ in range(k):
            lg, d_cache = _step(draft, d_cache, d_in, d_len, dev, all_logits=False)
            d_len += len(d_in)
            tok = int(lg[-1].argmax())
            drafts.append(tok)
            d_in = [tok]
        _sync(dev)
        tb = time.perf_counter()
        # ---------------- 2) verify all drafts with ONE target pass
        pending = seq[t_len:]                     # normally exactly one token
        lg, t_cache = _step(target, t_cache, pending + drafts, t_len, dev, all_logits=True)
        t_len += len(pending) + len(drafts)
        preds = lg[len(pending) - 1:].argmax(-1).tolist()   # k+1 target predictions
        _sync(dev)
        tc = time.perf_counter()
        draft_time += tb - ta
        verify_time += tc - tb
        # ---------------- 3) accept longest matching prefix, 4) correction / bonus token
        n = 0
        while n < k and drafts[n] == preds[n]:
            n += 1
        drafted += k
        accepted += n
        accept_hist[n] += 1
        new = drafts[:n] + [preds[n]]
        # respect EOS and the token budget
        remaining = max_new_tokens - len(gen)
        new = new[:remaining]
        if not ignore_eos:
            for j, t in enumerate(new):
                if t in eos:
                    new = new[: j + 1]
                    done = True
                    break
        seq += new
        gen += new
        if len(gen) >= max_new_tokens:
            done = True
        # ---------------- 5) roll back caches to len(seq) - 1 (last token stays pending)
        keep = len(seq) - 1
        if t_len > keep:
            t_cache.crop(keep)
            t_len = keep
        if d_len > keep:
            d_cache.crop(keep)
            d_len = keep
        assert t_cache.get_seq_length() == t_len, "target cache out of sync"
        assert d_cache.get_seq_length() == d_len, "draft cache out of sync"

    _sync(dev)
    t1 = time.perf_counter()
    extra = {"mode": f"speculative-k{k}", "k": k, "rounds": rounds, "drafted": drafted, "accepted": accepted,
             "acceptance_rate": accepted / drafted if drafted else 0.0,
             "mean_accepted_per_round": accepted / rounds if rounds else 0.0,
             "tokens_per_target_call": (len(gen) - 1) / rounds if rounds else 0.0,
             "accept_hist": accept_hist, "draft_time_s": draft_time, "verify_time_s": verify_time}
    return DecodeResult(gen, t_first - t0, t1 - t_first, t1 - t0, _peak_mem(dev), [], extra)
