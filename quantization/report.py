"""Quick sanity report for a freshly quantized model.

This is *not* a benchmark (EvalScope does that, see ``run_eval.py``). It checks
that the quantized model still produces coherent text and records:

* on-disk checkpoint size of the base and quantized model (+ compression ratio),
* peak GPU memory during a short generation,
* generation latency / tokens-per-second for one greedy prompt.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Explain in two sentences what model quantization is."
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def checkpoint_size_bytes(path: str | os.PathLike) -> int | None:
    """Total size of weight files in a local checkpoint dir, or ``None`` if not local."""
    p = Path(os.path.expanduser(path))
    if not p.is_dir():
        return None
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and f.suffix in WEIGHT_SUFFIXES)


def _mb(n: int | None) -> float | None:
    return None if n is None else round(n / 1024**2, 1)


def sample_generation(
    model: Any, tokenizer: Any, prompt: str = DEFAULT_PROMPT, max_new_tokens: int = 64
) -> dict[str, Any]:
    """Greedy-generate a short answer and time it."""
    import torch

    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return {
        "prompt": prompt,
        "output": tokenizer.decode(new_tokens, skip_special_tokens=True),
        "new_tokens": int(new_tokens.numel()),
        "latency_s": round(elapsed, 3),
        "tokens_per_s": round(new_tokens.numel() / elapsed, 2) if elapsed > 0 else None,
        "peak_gpu_mem_mb": _mb(torch.cuda.max_memory_allocated()) if cuda else None,
    }


def build_report(
    model: Any, tokenizer: Any, *, quantized_dir: str | os.PathLike, base_dir: str | os.PathLike
) -> dict[str, Any]:
    """Assemble size + generation stats into a JSON-serialisable dict."""
    base = checkpoint_size_bytes(base_dir)
    quant = checkpoint_size_bytes(quantized_dir)
    report: dict[str, Any] = {
        "base_size_mb": _mb(base),
        "quantized_size_mb": _mb(quant),
        "compression_ratio": round(base / quant, 2) if base and quant else None,
    }
    try:
        report["sample"] = sample_generation(model, tokenizer)
    except Exception as exc:  # noqa: BLE001 - report must never break the save
        logger.warning("Sample generation failed: %s", exc)
        report["sample"] = {"error": str(exc)}
    return report
