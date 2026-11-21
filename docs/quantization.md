# Model Quantization with torchao

This module applies **post-training quantization (PTQ)** to the project's LLM
(Qwen2.5-1.5B-Instruct by default) using [torchao](https://github.com/pytorch/ao),
saves the result as a regular Hugging Face checkpoint, and lets the existing
vLLM + EvalScope pipeline serve and benchmark it against the full-precision baseline.

```
 base model (BF16)                quantized checkpoint               benchmark
 ─────────────────  run_quantize  ────────────────────  vLLM +      ─────────────
 config.MODEL_ID   ─────────────► QUANT_OUTPUT_DIR     ─ EvalScope ► eval_outputs/
                   (torchao via   + quantization_meta.json            <model>-<scheme>/
                    transformers)
```

---

## 1. Quick start

```bash
uv sync                                   # installs torchao (added to pyproject.toml)

python run_quantize.py --list             # show schemes and their min GPU requirement
python run_quantize.py                    # quantize using the values in config.py
```

To **benchmark** the quantized model, set in `config.py`:

```python
QUANT_SCHEME = "int8_weight_only"
SERVE_QUANTIZED = True
```

and run the usual pipeline:

```bash
python run_pipeline.py
```

If the checkpoint does not exist yet, `run_pipeline.py` creates it first (in a
separate process so GPU memory is freed before vLLM starts), then serves it with
`--quantization torchao` and evaluates it. Results go to
`eval_outputs/<served-model-name>-<scheme>/`, so they never mix with the baseline
results.

---

## 2. Supported schemes

| Scheme (`QUANT_SCHEME`)              | Weights | Activations | Size vs BF16 | Min GPU | Notes |
|--------------------------------------|---------|-------------|--------------|---------|-------|
| `int8_weight_only`                   | INT8    | BF16        | ~½           | any / CPU | Near-lossless. **Recommended default.** |
| `int4_weight_only`                   | INT4 (group-wise) | BF16 | ~¼     | sm80 (Ampere) | Small accuracy loss; tune `QUANT_GROUP_SIZE`. |
| `int8_dynamic_act_int8_weight`       | INT8    | INT8 (dynamic, per token) | ~½ | sm75 | Speeds up compute-bound, large-batch serving. |
| `float8_dynamic_act_float8_weight`   | FP8 e4m3 (per row) | FP8 (dynamic) | ~½ | sm89 (Ada/Hopper) | **Not available on Ampere (A100 / RTX 30xx).** |

Only `nn.Linear` layers are quantized; embeddings, norms and `lm_head` stay in
`QUANT_DTYPE` (BF16 by default).

### Weight-only vs. dynamic activation quantization

* **Weight-only (W8A16 / W4A16):** weights are stored in low precision and
  dequantized on the fly inside the matmul kernel. This cuts memory and memory
  bandwidth, which is the bottleneck for small-batch / decode-heavy LLM
  inference. Activations keep full precision, so accuracy loss is small.
* **Dynamic activation (W8A8, FP8):** activations are also quantized at runtime
  (scales computed per token, no calibration data needed), so the matmul itself
  runs in INT8/FP8 on tensor cores. This helps most when throughput is
  compute-bound (large batches, long prefills).

### INT4 group size

INT4 stores one scale/zero-point per `group_size` consecutive weights.
Smaller groups follow the weight distribution more closely (better accuracy)
at the cost of more metadata. `128` is a good default; try `64` or `32` if
accuracy drops noticeably.

---

## 3. Hardware checks

Before loading anything, `quantization/hardware.py` compares the scheme's
requirements with the visible GPU (`config.GPU_ID`). For example, on an A100:

```
[QUANTIZE] ERROR: Scheme 'float8_dynamic_act_float8_weight' needs compute capability >= 8.9,
but 'NVIDIA A100-SXM4-40GB' has 8.0. On Ampere GPUs use an INT8 or INT4 scheme instead.
```

---

## 4. Configuration reference (`config.py`)

| Setting             | Default                          | Meaning |
|---------------------|----------------------------------|---------|
| `QUANT_SCHEME`      | `"int8_weight_only"`             | Scheme to apply (see table above). |
| `QUANT_GROUP_SIZE`  | `128`                            | Group size for `int4_weight_only`. |
| `QUANT_DTYPE`       | `"bfloat16"`                     | Dtype of non-quantized layers and activations. |
| `QUANT_OUTPUT_ROOT` | `~/nxp/models/LLMs/quantized`    | Parent folder for quantized checkpoints. |
| `QUANT_OUTPUT_DIR`  | `<root>/<served-name>-<scheme>`  | Derived; one folder per scheme. |
| `SERVE_QUANTIZED`   | `False`                          | Serve/evaluate the quantized checkpoint instead of the base model. |

When `SERVE_QUANTIZED = True`, `config.py` automatically:

* keeps the original path in `BASE_MODEL_ID`,
* points `MODEL_ID` at `QUANT_OUTPUT_DIR`,
* appends the scheme to `SERVED_MODEL_NAME` (e.g. `qwen2.5-1.5b-instruct-int8_weight_only`),
* sets `VLLM_QUANTIZATION = "torchao"`, which `start_server.sh` passes on as `--quantization torchao`,
* sets `WORK_DIR = ./eval_outputs/<SERVED_MODEL_NAME>`.

## 5. CLI reference (`run_quantize.py`)

All flags default to the `config.py` values.

| Flag             | Description |
|------------------|-------------|
| `--list`         | Print available schemes and exit. |
| `--model`        | Base model path or Hub id (default `BASE_MODEL_ID`). |
| `--scheme`       | Scheme name. |
| `--output-dir`   | Output folder (default derived from `QUANT_OUTPUT_ROOT` + scheme). |
| `--group-size`   | INT4 group size. |
| `--dtype`        | `bfloat16` or `float16`. |
| `--overwrite`    | Replace an existing output folder. |
| `--no-report`    | Skip the sanity report (sample generation). |

Example – produce all Ampere-compatible variants for a comparison table:

```bash
for s in int8_weight_only int4_weight_only int8_dynamic_act_int8_weight; do
    python run_quantize.py --scheme "$s" --overwrite
done
```

---

## 6. Output

Each output folder is a standard Hugging Face checkpoint:

```
qwen2.5-1.5b-instruct-int8_weight_only/
├── config.json               # includes "quantization_config" (torchao)
├── model.safetensors         # or pytorch_model.bin on older torchao/transformers
├── tokenizer*.json, ...
└── quantization_meta.json    # written by this module
```

`quantization_meta.json` records how the checkpoint was produced plus a quick
sanity report:

```json
{
  "base_model": "~/nxp/models/LLMs/Qwen/Qwen2.5-1.5B-Instruct",
  "scheme": "int8_weight_only",
  "options": {"group_size": 128, "int4_packing_format": "tile_packed_to_4d", "extra": {}},
  "serialization": "safetensors",
  "device": {"device": "cuda", "name": "NVIDIA A100", "compute_capability": [8, 0]},
  "versions": {"torch": "...", "torchao": "...", "transformers": "..."},
  "report": {
    "base_size_mb": 2944.4,
    "quantized_size_mb": 1720.3,
    "compression_ratio": 1.71,
    "sample": {"output": "...", "tokens_per_s": 21.4, "peak_gpu_mem_mb": 2105.0}
  }
}
```

(Numbers above are illustrative.) The compression ratio is below the
theoretical 2× / 4× because embeddings and `lm_head` (a large share of a 1.5B
model with a 152k vocabulary) are not quantized.

> The sanity report is **not** a benchmark: it only shows the model still
> produces coherent text. Note that the tokens/s measured with plain
> `transformers.generate` (no `torch.compile`) can be *lower* than BF16 for
> some schemes. Use the vLLM + EvalScope pipeline for accuracy and the
> `evalscope perf` tooling for throughput.

---

## 7. Code structure

```
quantization/
├── __init__.py    # public API: QuantizationJob, quantize_and_save, list_schemes, ...
├── schemes.py     # scheme registry -> torchao AOBaseConfig builders
├── hardware.py    # GPU detection + per-scheme capability checks
├── quantizer.py   # load (TorchAoConfig) -> save -> metadata
└── report.py      # checkpoint size, sample generation, latency
run_quantize.py    # CLI wrapper reading defaults from config.py
tests/test_quantization.py  # registry + hardware-check tests (no GPU needed)
```

### Using it from Python

```python
from quantization import QuantizationJob, QuantOptions, quantize_and_save

meta = quantize_and_save(QuantizationJob(
    model_path="~/nxp/models/LLMs/Qwen/Qwen2.5-1.5B-Instruct",
    output_dir="/tmp/qwen-int4-g64",
    scheme="int4_weight_only",
    options=QuantOptions(group_size=64),
))
print(meta["report"]["compression_ratio"])
```

### Adding a new scheme

Register a builder in `quantization/schemes.py`; nothing else changes and the
new name immediately works in `config.py`, the CLI and the pipeline:

```python
@register_scheme(
    "int8_dynamic_act_int4_weight",
    "INT4 weights + dynamic INT8 activations (W4A8).",
    min_compute_capability=(8, 0),
)
def _int8_dyn_int4(opts: QuantOptions):
    from torchao.quantization import Int8DynamicActivationInt4WeightConfig
    return Int8DynamicActivationInt4WeightConfig(group_size=opts.group_size)
```

`QuantOptions.extra` forwards arbitrary keyword arguments to the torchao config
for one-off experiments.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `needs compute capability >= 8.9` | FP8 on Ampere. Use an INT8/INT4 scheme. |
| `Output directory ... is not empty` | Add `--overwrite`. |
| `safetensors save failed ... falling back` | Older torchao/transformers; the `.bin` checkpoint still works with vLLM. |
| vLLM fails to load an `int4_weight_only` checkpoint | vLLM's torchao support depends on the INT4 packing format and versions. INT8 schemes are the most reliable path; try upgrading vLLM/torchao, or change `QuantOptions.int4_packing_format`. |
| `ModuleNotFoundError: torchao` | Run `uv sync` (torchao is now in `pyproject.toml`). |
| vLLM: "not enough memory for KV cache" | Quantization shrinks weights, but the KV cache must still fit in `GPU_MEM_UTILIZATION`. Reduce `MAX_MODEL_LEN` or raise `GPU_MEM_UTILIZATION`. |
