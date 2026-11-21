"""CLI entry point: quantize a model with torchao and save a vLLM-servable checkpoint.

Defaults come from ``config.py``; every value can be overridden on the command line.

Examples::

    python run_quantize.py --list                          # show available schemes
    python run_quantize.py                                 # use config.py values
    python run_quantize.py --scheme int4_weight_only --group-size 64 --overwrite

See docs/quantization.md for details.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import config

# Pin GPU and CPU threads before torch is imported (same policy as the server).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.GPU_ID)
os.environ["OMP_NUM_THREADS"] = config.CPU_THREADS
os.environ["MKL_NUM_THREADS"] = config.CPU_THREADS

from quantization import QuantizationJob, QuantOptions, list_schemes, quantize_and_save  # noqa: E402
from quantization.hardware import UnsupportedHardwareError  # noqa: E402


def default_output_dir(scheme: str) -> str:
    """Output dir for ``scheme``: config value if it matches, else derived from the root."""
    if scheme == config.QUANT_SCHEME:
        return config.QUANT_OUTPUT_DIR
    base_name = config.SERVED_MODEL_NAME.removesuffix(f"-{config.QUANT_SCHEME}")
    return os.path.join(config.QUANT_OUTPUT_ROOT, f"{base_name}-{scheme}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantize a Hugging Face model with torchao.")
    p.add_argument("--list", action="store_true", help="List available schemes and exit.")
    p.add_argument("--model", default=config.BASE_MODEL_ID, help="Base model path or Hub id.")
    p.add_argument("--scheme", default=config.QUANT_SCHEME, help="Quantization scheme name.")
    p.add_argument("--output-dir", default=None, help="Where to save the quantized checkpoint.")
    p.add_argument("--group-size", type=int, default=config.QUANT_GROUP_SIZE,
                   help="Group size for group-wise schemes (int4).")
    p.add_argument("--dtype", default=config.QUANT_DTYPE, choices=["bfloat16", "float16"],
                   help="Dtype for non-quantized parts and activations.")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing output dir.")
    p.add_argument("--no-report", action="store_true", help="Skip the sanity report.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[QUANTIZE] %(levelname)s %(message)s")
    args = parse_args(argv)

    if args.list:
        for s in list_schemes():
            cc = ".".join(map(str, s.min_compute_capability)) if s.min_compute_capability else "-"
            print(f"{s.name:<36} min_sm={cc:<5} {s.description}")
        return 0

    job = QuantizationJob(
        model_path=args.model,
        output_dir=args.output_dir or default_output_dir(args.scheme),
        scheme=args.scheme,
        options=QuantOptions(group_size=args.group_size),
        dtype=args.dtype,
        overwrite=args.overwrite,
    )

    try:
        meta = quantize_and_save(job, run_report=not args.no_report)
    except (UnsupportedHardwareError, FileExistsError, ValueError) as exc:
        print(f"[QUANTIZE] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\n[QUANTIZE] Done -> {job.resolved_output_dir()}")
    print(json.dumps(meta.get("report", {}), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
