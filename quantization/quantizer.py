"""Load -> quantize -> save pipeline built on transformers' ``TorchAoConfig``.

Quantization happens at load time: transformers loads the BF16 weights and
replaces every ``nn.Linear`` (except ``lm_head``) with a torchao quantized
tensor. The result is saved as a regular Hugging Face checkpoint whose
``config.json`` carries a ``quantization_config`` entry, which vLLM recognises
(``--quantization torchao``).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quantization.hardware import check_scheme_supported
from quantization.schemes import QuantOptions, get_scheme

logger = logging.getLogger(__name__)

METADATA_FILENAME = "quantization_meta.json"


@dataclass
class QuantizationJob:
    """Everything needed to produce one quantized checkpoint.

    Attributes:
        model_path: Local path (``~`` allowed) or Hub id of the BF16/FP16 model.
        output_dir: Where the quantized checkpoint is written.
        scheme: Name of a registered scheme (see ``quantization.schemes``).
        options: Scheme options (group size, ...).
        dtype: Dtype for the non-quantized parts and activations.
        overwrite: Replace ``output_dir`` if it already exists.
        trust_remote_code: Forwarded to ``from_pretrained``.
    """

    model_path: str
    output_dir: str
    scheme: str = "int8_weight_only"
    options: QuantOptions = field(default_factory=QuantOptions)
    dtype: str = "bfloat16"
    overwrite: bool = False
    trust_remote_code: bool = True

    def resolved_model_path(self) -> str:
        expanded = os.path.expanduser(self.model_path)
        return expanded if os.path.exists(expanded) else self.model_path

    def resolved_output_dir(self) -> Path:
        return Path(os.path.expanduser(self.output_dir))


def is_quantized_checkpoint(path: str | os.PathLike) -> bool:
    """True if ``path`` contains a checkpoint produced by this package."""
    return (Path(os.path.expanduser(path)) / METADATA_FILENAME).is_file()


def _prepare_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory '{out_dir}' is not empty. Use --overwrite to replace it."
            )
        logger.warning("Removing existing output directory %s", out_dir)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def load_quantized_model(job: QuantizationJob, device: str) -> tuple[Any, Any]:
    """Load the base model and quantize it on the fly. Returns ``(model, tokenizer)``."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

    scheme = get_scheme(job.scheme)
    ao_config = scheme.build(job.options)
    logger.info("torchao config: %r", ao_config)

    model_path = job.resolved_model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=job.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=getattr(torch, job.dtype),
        device_map=device,
        quantization_config=TorchAoConfig(quant_type=ao_config),
        trust_remote_code=job.trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def save_quantized_model(model: Any, tokenizer: Any, out_dir: Path) -> str:
    """Save model + tokenizer. Returns the serialization format actually used.

    Recent torchao/transformers can write torchao tensors to safetensors. Older
    combinations only support pickle-based ``.bin`` files, so we fall back.
    """
    try:
        model.save_pretrained(out_dir)
        fmt = "safetensors"
    except Exception as exc:  # noqa: BLE001 - library raises various types here
        logger.warning("safetensors save failed (%s); falling back to pytorch .bin", exc)
        for f in out_dir.glob("*.safetensors*"):
            f.unlink()
        model.save_pretrained(out_dir, safe_serialization=False)
        fmt = "pytorch_bin"
    tokenizer.save_pretrained(out_dir)
    return fmt


def quantize_and_save(job: QuantizationJob, *, run_report: bool = True) -> dict[str, Any]:
    """Run the full pipeline and return the metadata written next to the checkpoint.

    Steps: hardware check -> load + quantize -> save -> (optional) sanity report.
    """
    scheme = get_scheme(job.scheme)
    device_info = check_scheme_supported(scheme)
    out_dir = job.resolved_output_dir()
    _prepare_output_dir(out_dir, job.overwrite)

    logger.info(
        "Quantizing '%s' with scheme '%s' on %s", job.model_path, scheme.name, device_info.name
    )
    model, tokenizer = load_quantized_model(job, device_info.device)
    fmt = save_quantized_model(model, tokenizer, out_dir)
    logger.info("Saved quantized checkpoint (%s) to %s", fmt, out_dir)

    import torch
    import torchao
    import transformers

    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model": job.model_path,
        "scheme": scheme.name,
        "scheme_description": scheme.description,
        "options": asdict(job.options),
        "dtype": job.dtype,
        "serialization": fmt,
        "device": asdict(device_info),
        "versions": {
            "torch": torch.__version__,
            "torchao": torchao.__version__,
            "transformers": transformers.__version__,
        },
    }

    if run_report:
        from quantization.report import build_report

        metadata["report"] = build_report(
            model, tokenizer, quantized_dir=out_dir, base_dir=job.resolved_model_path()
        )

    with open(out_dir / METADATA_FILENAME, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata
