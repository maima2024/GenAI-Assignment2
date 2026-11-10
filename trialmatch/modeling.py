"""Model / tokenizer loading and small helpers shared by training, evaluation, decoders and the app."""
from __future__ import annotations

import json
import os
import platform

import torch

TARGET_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_device(device: str | None = None) -> str:
    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(dtype: str | None, device: str) -> torch.dtype:
    if dtype in (None, "auto"):
        return torch.float16 if device.startswith("cuda") else torch.float32
    return {"fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16, "fp32": torch.float32, "float32": torch.float32}[dtype]


def _dtype_kwargs(dt):
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dt} if (major, minor) >= (4, 56) else {"torch_dtype": dt}


def load_tokenizer(name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name: str, adapter: str | None = None, dtype: str | None = "auto",
               device: str | None = "auto", attn_implementation: str = "sdpa", merge: bool = True):
    """Load a causal LM, optionally apply a LoRA adapter (local dir or HF Hub id) and merge it."""
    from transformers import AutoModelForCausalLM
    device = pick_device(device)
    dt = pick_dtype(dtype, device)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation=attn_implementation,
                                                 **_dtype_kwargs(dt))
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            model = model.merge_and_unload()
    model.to(device)
    model.eval()
    return model


def eos_ids(tokenizer, model=None) -> list[int]:
    ids = set()
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            ids.add(tid)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return sorted(ids)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def model_card(model) -> dict:
    c = model.config
    nh = c.num_attention_heads
    return {"params": count_params(model), "layers": c.num_hidden_layers, "hidden": c.hidden_size,
            "heads": nh, "kv_heads": getattr(c, "num_key_value_heads", nh),
            "head_dim": getattr(c, "head_dim", None) or c.hidden_size // nh,
            "intermediate": getattr(c, "intermediate_size", None), "vocab": c.vocab_size,
            "max_position_embeddings": getattr(c, "max_position_embeddings", None),
            "rope_theta": rope_theta(c), "tie_word_embeddings": getattr(c, "tie_word_embeddings", None)}


def rope_theta(config) -> float:
    th = getattr(config, "rope_theta", None)
    if th is None:
        rp = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        th = rp.get("rope_theta", 10000.0)
    return float(th)


def env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "cpu_count": os.cpu_count()}
    try:
        import transformers, peft, accelerate
        info.update(transformers=transformers.__version__, peft=peft.__version__,
                    accelerate=accelerate.__version__)
    except Exception as e:  # pragma: no cover
        info["lib_error"] = str(e)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 2**30, 2),
                    cuda=torch.version.cuda, gpu_count=torch.cuda.device_count(),
                    compute_capability=f"{p.major}.{p.minor}")
    return info


if __name__ == "__main__":
    print(json.dumps(env_info(), indent=2))
