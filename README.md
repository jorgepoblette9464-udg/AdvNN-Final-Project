# AdvNN Final Project – LLM Benchmarking & Quantization

Serve an LLM (default: **Qwen2.5-1.5B-Instruct**) with **vLLM**, benchmark it with
**EvalScope** (MMLU, MMLU-Pro, HellaSwag), and compare it against
**torchao-quantized** variants of the same model.

## Project layout

| Path | Purpose |
|------|---------|
| `config.py` | Single source of truth: model path, GPU limits, datasets, quantization settings. |
| `start_server.sh` | Starts the vLLM OpenAI-compatible server using `config.py`. |
| `run_eval.py` | Runs EvalScope against the running server (resumable via `WORK_DIR`). |
| `run_pipeline.py` | Orchestrator: (quantize if needed) → start server → evaluate → shut down. |
| `run_quantize.py` | CLI to quantize a model with torchao. |
| `quantization/` | Modular quantization package (schemes, hardware checks, quantizer, report). |
| `docs/quantization.md` | Full quantization documentation. |
| `tests/` | Unit tests (`python -m pytest tests/`). |
| `eval_outputs/` | EvalScope predictions, reviews and reports. |

## Setup

```bash
uv sync
```

Requires a Linux machine with an NVIDIA GPU (vLLM does not run on Windows).

## Baseline benchmark

1. Edit `config.py` (`MODEL_ID`, `GPU_ID`, `DATASETS`, `WORK_DIR`, ...).
2. Run:

```bash
python run_pipeline.py
```

## Quantization (torchao)

Supported schemes: `int8_weight_only` (default), `int4_weight_only`,
`int8_dynamic_act_int8_weight`, and `float8_dynamic_act_float8_weight` (Ada/Hopper only).

```bash
python run_quantize.py --list                             # show schemes
python run_quantize.py                                    # quantize with config.py values
python run_quantize.py --scheme int4_weight_only --group-size 64 --overwrite
```

To benchmark a quantized model, set `QUANT_SCHEME` and `SERVE_QUANTIZED = True` in
`config.py` and run `python run_pipeline.py`. The checkpoint is created automatically
if it is missing, served by vLLM with `--quantization torchao`, and evaluated into
`eval_outputs/<model>-<scheme>/`.

See **[docs/quantization.md](docs/quantization.md)** for the scheme comparison,
configuration/CLI reference, output format, how to add new schemes, and troubleshooting.
