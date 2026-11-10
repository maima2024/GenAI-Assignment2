"""Greedy autoregressive decoding with and without KV-cache reuse.

Three decoders are provided:

1. greedy_uncached   -- REFERENCE.  Every step re-runs the full model on the whole sequence
                        (prompt + generated so far), use_cache=False.  O(n^2) work per token.
2. greedy_cached     -- Cache-aware loop over the Hugging Face model's forward().  We create the
                        DynamicCache ourselves, run one prefill pass over the prompt, then feed ONE
                        token per step together with explicit position_ids / cache_position and an
                        attention mask whose length equals (cached + new) tokens.  We assert the
                        cache grows by exactly one position per step.
3. ScratchDecoder    -- A from-scratch forward pass for Qwen2-family weights (RMSNorm, RoPE, GQA,
                        SwiGLU taken from the loaded modules) with our OWN pre-allocated KV cache
                        tensors and a pluggable attention kernel (reference / tiled online-softmax /
                        torch SDPA).  This exposes every step of cache-aware decoding: rotary
                        positions are computed for absolute positions [past, past+T), new K/V are
                        written into the cache slice, and the causal mask is bottom-right aligned.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .attention import ATTENTION_IMPLS
from .modeling import rope_theta

_KEEP_KW = {"name": "logits_to_keep"}


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def model_forward(model, keep_last: int | None = 1, **kw):
    """Call model(**kw) computing logits only for the last `keep_last` positions (0/None = all)."""
    k = 0 if not keep_last else keep_last
    name = _KEEP_KW["name"]
    if name:
        try:
            return model(**kw, **{name: k})
        except TypeError:
            _KEEP_KW["name"] = "num_logits_to_keep" if name == "logits_to_keep" else None
            return model_forward(model, keep_last, **kw)
    return model(**kw)


def new_dynamic_cache(model=None):
    from transformers import DynamicCache
    try:
        return DynamicCache(config=model.config) if model is not None else DynamicCache()
    except TypeError:
        return DynamicCache()


def kv_cache_bytes(config, seq_len: int, dtype: torch.dtype, batch: int = 1) -> int:
    nh = config.num_attention_heads
    nkv = getattr(config, "num_key_value_heads", nh)
    hd = getattr(config, "head_dim", None) or config.hidden_size // nh
    elt = torch.finfo(dtype).bits // 8
    return 2 * config.num_hidden_layers * batch * nkv * hd * seq_len * elt


@dataclass
class DecodeResult:
    tokens: list[int]
    prefill_s: float
    decode_s: float
    total_s: float
    peak_mem_bytes: int | None
    margins: list[float] = field(default_factory=list)  # top1 - top2 logit at each step
    extra: dict = field(default_factory=dict)

    @property
    def tokens_per_s(self) -> float:
        return len(self.tokens) / self.total_s if self.total_s > 0 else 0.0

    @property
    def decode_tokens_per_s(self) -> float:
        n = max(len(self.tokens) - 1, 0)
        return n / self.decode_s if self.decode_s > 0 else 0.0

    def stats(self) -> dict:
        return {"n_tokens": len(self.tokens), "prefill_s": self.prefill_s, "decode_s": self.decode_s,
                "latency_s": self.total_s, "tokens_per_s": self.tokens_per_s,
                "decode_tokens_per_s": self.decode_tokens_per_s,
                "peak_mem_mb": None if self.peak_mem_bytes is None else self.peak_mem_bytes / 2**20,
                **self.extra}


def _top2_margin(logits_row: torch.Tensor) -> float:
    t = torch.topk(logits_row.float(), 2).values
    return float(t[0] - t[1])


def _start_mem(device):
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_mem(device):
    return torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None


@torch.no_grad()
def greedy_uncached(model, input_ids: torch.Tensor, max_new_tokens: int, eos: list[int] | None = None,
                    ignore_eos: bool = False, record_margins: bool = False) -> DecodeResult:
    dev = input_ids.device
    _start_mem(dev)
    eos = set(eos or [])
    ids = input_ids
    toks, margins = [], []
    _sync(dev)
    t0 = time.perf_counter()
    t_first = None
    for _ in range(max_new_tokens):
        out = model_forward(model, 1, input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        row = out.logits[0, -1]
        nxt = int(row.argmax())
        if record_margins:
            margins.append(_top2_margin(row))
        toks.append(nxt)
        if t_first is None:
            _sync(dev)
            t_first = time.perf_counter()
        if nxt in eos and not ignore_eos:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
    _sync(dev)
    t1 = time.perf_counter()
    return DecodeResult(toks, t_first - t0, t1 - t_first, t1 - t0, _peak_mem(dev), margins,
                        {"mode": "uncached"})


@torch.no_grad()
def greedy_cached(model, input_ids: torch.Tensor, max_new_tokens: int, eos: list[int] | None = None,
                  ignore_eos: bool = False, record_margins: bool = False, check_cache: bool = True) -> DecodeResult:
    dev = input_ids.device
    _start_mem(dev)
    eos = set(eos or [])
    L = input_ids.shape[1]
    cache = new_dynamic_cache(model)
    toks, margins = [], []
    _sync(dev)
    t0 = time.perf_counter()
    # ---- prefill: all prompt positions at once, causal mask built by the model from the
    # 2-D padding mask + cache_position; K/V for positions 0..L-1 are written to the cache.
    pos = torch.arange(L, device=dev)
    out = model_forward(model, 1, input_ids=input_ids, attention_mask=torch.ones(1, L, dtype=torch.long, device=dev),
                        position_ids=pos[None], cache_position=pos, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    row = out.logits[0, -1]
    nxt = int(row.argmax())
    if record_margins:
        margins.append(_top2_margin(row))
    toks.append(nxt)
    _sync(dev)
    t_first = time.perf_counter()
    cur = L  # number of positions stored in the cache
    lengths = [cache.get_seq_length()] if check_cache else []
    while len(toks) < max_new_tokens and (ignore_eos or nxt not in eos):
        # ---- decode step: ONE new token at absolute position `cur`; it may attend to all
        # cached positions 0..cur-1 and itself, so the 2-D mask has length cur+1 (no future keys exist).
        step_pos = torch.tensor([cur], device=dev)
        out = model_forward(model, 1, input_ids=torch.tensor([[nxt]], device=dev),
                            attention_mask=torch.ones(1, cur + 1, dtype=torch.long, device=dev),
                            position_ids=step_pos[None], cache_position=step_pos,
                            past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        cur += 1
        if check_cache:
            n = cache.get_seq_length()
            assert n == cur, f"cache length {n} != expected {cur}"
            lengths.append(n)
        row = out.logits[0, -1]
        nxt = int(row.argmax())
        if record_margins:
            margins.append(_top2_margin(row))
        toks.append(nxt)
    _sync(dev)
    t1 = time.perf_counter()
    cfg = model.config
    dt = next(model.parameters()).dtype
    return DecodeResult(toks, t_first - t0, t1 - t_first, t1 - t0, _peak_mem(dev), margins,
                        {"mode": "cached", "final_cache_len": cur,
                         "kv_cache_mb_analytic": kv_cache_bytes(cfg, cur, dt) / 2**20,
                         "cache_len_trace_ok": bool(lengths) and lengths == list(range(L, cur + 1))})


# =============================================================================== scratch
class ScratchKVCache:
    """Pre-allocated per-layer K/V buffers of shape (B, n_kv_heads, max_len, head_dim)."""

    def __init__(self, n_layers, batch, n_kv, max_len, head_dim, device, dtype):
        self.k = [torch.zeros(batch, n_kv, max_len, head_dim, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(batch, n_kv, max_len, head_dim, device=device, dtype=dtype) for _ in range(n_layers)]
        self.len = 0
        self.max_len = max_len

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


class ScratchDecoder:
    """From-scratch Qwen2 forward pass with explicit KV cache and pluggable attention kernel."""

    def __init__(self, hf_model, attention: str = "reference", block: int = 64):
        self.hf = hf_model
        self.base = hf_model.model
        cfg = hf_model.config
        self.nh = cfg.num_attention_heads
        self.nkv = getattr(cfg, "num_key_value_heads", self.nh)
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.nh
        self.n_layers = cfg.num_hidden_layers
        self.attn_name = attention
        self.block = block
        theta = rope_theta(cfg)
        dev = next(hf_model.parameters()).device
        self.device = dev
        self.dtype = next(hf_model.parameters()).dtype
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.hd, 2, device=dev, dtype=torch.float32) / self.hd))

    def _attn(self, q, k, v):
        fn = ATTENTION_IMPLS[self.attn_name]
        if self.attn_name == "tiled":
            return fn(q, k, v, causal=True, block_q=self.block, block_k=self.block)
        return fn(q, k, v, causal=True)

    def _rope(self, positions: torch.Tensor):
        freqs = positions.float()[:, None] * self.inv_freq[None, :]      # (T, hd/2)
        emb = torch.cat([freqs, freqs], dim=-1)                           # (T, hd)
        return emb.cos().to(self.dtype)[None, None], emb.sin().to(self.dtype)[None, None]

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def new_cache(self, max_len: int, batch: int = 1) -> ScratchKVCache:
        return ScratchKVCache(self.n_layers, batch, self.nkv, max_len, self.hd, self.device, self.dtype)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, cache: ScratchKVCache | None = None) -> torch.Tensor:
        """ids (B, T) -> logits of the LAST position (B, V).  With a cache, ids are the NEW tokens
        at absolute positions [cache.len, cache.len + T)."""
        B, T = ids.shape
        past = cache.len if cache is not None else 0
        if cache is not None and past + T > cache.max_len:
            raise ValueError("scratch cache overflow")
        cos, sin = self._rope(torch.arange(past, past + T, device=ids.device))
        h = self.base.embed_tokens(ids)
        rep = self.nh // self.nkv
        for i, layer in enumerate(self.base.layers):
            a = layer.self_attn
            x = layer.input_layernorm(h)
            q = a.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
            k = a.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
            v = a.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
            q = q * cos + self._rotate_half(q) * sin       # RoPE at absolute positions
            k = k * cos + self._rotate_half(k) * sin
            if cache is not None:
                cache.k[i][:, :, past:past + T] = k        # write new keys/values into the cache
                cache.v[i][:, :, past:past + T] = v
                k = cache.k[i][:, :, :past + T]           # read all keys/values seen so far
                v = cache.v[i][:, :, :past + T]
            k = k.repeat_interleave(rep, dim=1)            # grouped-query attention
            v = v.repeat_interleave(rep, dim=1)
            o = self._attn(q, k, v)                        # bottom-right causal mask (Lk >= Lq)
            o = o.transpose(1, 2).reshape(B, T, self.nh * self.hd)
            h = h + a.o_proj(o)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        if cache is not None:
            cache.len += T
        h = self.base.norm(h[:, -1:])
        return self.hf.lm_head(h)[:, -1]

    @torch.no_grad()
    def greedy(self, input_ids, max_new_tokens, eos=None, ignore_eos=False, use_cache=True,
               record_margins=False) -> DecodeResult:
        dev = input_ids.device
        _start_mem(dev)
        eos = set(eos or [])
        toks, margins = [], []
        cache = self.new_cache(input_ids.shape[1] + max_new_tokens) if use_cache else None
        _sync(dev)
        t0 = time.perf_counter()
        logits = self.forward(input_ids, cache)
        seq = input_ids
        t_first = None
        while True:
            row = logits[0]
            nxt = int(row.argmax())
            if record_margins:
                margins.append(_top2_margin(row))
            toks.append(nxt)
            if t_first is None:
                _sync(dev)
                t_first = time.perf_counter()
            if len(toks) >= max_new_tokens or (nxt in eos and not ignore_eos):
                break
            new = torch.tensor([[nxt]], device=dev)
            if use_cache:
                logits = self.forward(new, cache)
            else:
                seq = torch.cat([seq, new], dim=1)
                logits = self.forward(seq, None)
        _sync(dev)
        t1 = time.perf_counter()
        return DecodeResult(toks, t_first - t0, t1 - t_first, t1 - t0, _peak_mem(dev), margins,
                            {"mode": f"scratch-{'cached' if use_cache else 'uncached'}-{self.attn_name}",
                             "scratch_cache_mb": cache.nbytes() / 2**20 if cache else 0})


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))
