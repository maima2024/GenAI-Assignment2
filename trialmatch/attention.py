"""Reference scaled-dot-product attention and a tiled attention with online softmax.

Shapes: q (B, H, Lq, D), k/v (B, H, Lk, D).  Causal masking is "bottom-right aligned":
query i sits at absolute position (Lk - Lq) + i and may attend to keys j <= (Lk - Lq) + i.
This is the alignment needed for KV-cached decoding (Lq new tokens after Lk - Lq cached ones).
NOTE: torch.nn.functional.scaled_dot_product_attention(is_causal=True) uses *top-left*
alignment, so it only matches when Lq == Lk (or with an explicit mask).
"""
from __future__ import annotations

import math

import torch


def causal_mask(Lq: int, Lk: int, device) -> torch.Tensor:
    """Boolean (Lq, Lk) mask, True where attention is NOT allowed."""
    off = Lk - Lq
    qi = torch.arange(Lq, device=device)[:, None] + off
    kj = torch.arange(Lk, device=device)[None, :]
    return kj > qi


def reference_attention(q, k, v, causal: bool = True, scale: float | None = None) -> torch.Tensor:
    """Materialises the full (Lq x Lk) score matrix; softmax computed stably in fp32
    by subtracting the row maximum before exponentiation."""
    D = q.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    if causal:
        assert k.shape[-2] >= q.shape[-2], "causal attention needs Lk >= Lq"
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if causal:
        s = s.masked_fill(causal_mask(q.shape[-2], k.shape[-2], q.device), float("-inf"))
    m = s.amax(dim=-1, keepdim=True)
    p = torch.exp(s - m)
    p = p / p.sum(dim=-1, keepdim=True)
    return torch.matmul(p, v.float()).to(q.dtype)


def tiled_attention(q, k, v, causal: bool = True, scale: float | None = None,
                    block_q: int = 64, block_k: int = 64) -> torch.Tensor:
    """FlashAttention-style blocked attention in plain PyTorch.

    For each query block we stream over key/value blocks and keep, per query row,
      m  = running maximum of scores seen so far,
      l  = running sum of exp(score - m),
      acc= running sum of exp(score - m) * v.
    When a new block raises the maximum from m to m', previous l and acc are rescaled by
    exp(m - m').  The output is acc / l.  Peak extra memory is O(block_q * block_k) per
    (batch, head) instead of O(Lq * Lk).  Key blocks entirely above the causal diagonal are
    skipped; only diagonal blocks need an explicit mask.
    """
    B, H, Lq, D = q.shape
    Lk = k.shape[-2]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    off = Lk - Lq
    if causal:
        assert off >= 0, "causal attention needs Lk >= Lq"
    out = torch.empty_like(q)
    acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    for qs in range(0, Lq, block_q):
        qe = min(qs + block_q, Lq)
        qb = q[:, :, qs:qe].to(acc_dtype) * scale
        nq = qe - qs
        m = torch.full((B, H, nq, 1), float("-inf"), device=q.device, dtype=acc_dtype)
        l = torch.zeros((B, H, nq, 1), device=q.device, dtype=acc_dtype)
        acc = torch.zeros((B, H, nq, D), device=q.device, dtype=acc_dtype)
        k_stop = min(Lk, qe - 1 + off + 1) if causal else Lk  # last key the last query may see
        for ks in range(0, k_stop, block_k):
            ke = min(ks + block_k, k_stop)
            s = torch.matmul(qb, k[:, :, ks:ke].to(acc_dtype).transpose(-1, -2))
            if causal and (ke - 1) > (qs + off):  # block crosses the diagonal -> partial mask
                qi = torch.arange(qs, qe, device=q.device)[:, None] + off
                kj = torch.arange(ks, ke, device=q.device)[None, :]
                s = s.masked_fill(kj > qi, float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            p = torch.exp(s - m_safe)
            alpha = torch.exp(m - m_safe)
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            acc = alpha * acc + torch.matmul(p, v[:, :, ks:ke].to(acc_dtype))
            m = m_new
        out[:, :, qs:qe] = (acc / l.clamp_min(torch.finfo(acc_dtype).tiny)).to(q.dtype)
    return out


def torch_sdpa(q, k, v, causal: bool = True, scale: float | None = None) -> torch.Tensor:
    """PyTorch fused SDPA with an explicit bottom-right causal mask when Lq != Lk."""
    import torch.nn.functional as F
    Lq, Lk = q.shape[-2], k.shape[-2]
    if causal and Lq == Lk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    mask = None
    if causal:
        mask = ~causal_mask(Lq, Lk, q.device)  # True = keep
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


ATTENTION_IMPLS = {"reference": reference_attention, "tiled": tiled_attention, "sdpa": torch_sdpa}
