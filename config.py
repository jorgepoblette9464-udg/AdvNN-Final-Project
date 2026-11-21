import os

# Model & Server Settings
MODEL_ID = "~/nxp/models/LLMs/Qwen/Qwen2.5-1.5B-Instruct"
SERVED_MODEL_NAME = "qwen2.5-1.5b-instruct"
HOST = "127.0.0.1"
PORT = 8801
API_URL = f"http://{HOST}:{PORT}/v1"

# Resource Limits (Shared Server Friendly)
GPU_ID = "0"                    # Target specific GPU index (e.g., "0" or "1")
GPU_MEM_UTILIZATION = 0.35      # Cap VRAM at 35% (~5-8GB on a 16-24GB card; 1.5B FP16 only needs ~3.5GB)
MAX_MODEL_LEN = 4096            # Restrict context window to save KV-cache VRAM
CPU_THREADS = "4"               # Limit PyTorch/OpenMP CPU threads so you don't choke shared CPUs

# Evaluation Settings
DATASETS = ["hellaswag"] #["mmlu", "mmlu_pro", "hellaswag"]
EVAL_BATCH_SIZE = 32            # Concurrent API requests sent to vLLM (adjust based on GPU_MEM_UTILIZATION)
WORK_DIR = "./eval_outputs/qwen2.5_1.5b_2"  # Fixed output folder required for resuming


# Shared decoding parameters passed directly to EvalScope TaskConfig
GENERATION_CONFIG = {
    "do_sample": False,          # Greedy decoding for deterministic results
    "temperature": 0.0,         # Greedy decoding for deterministic results
    "top_p": 1.0,
    "max_tokens": 512,          # Enough headroom for MMLU-Pro CoT reasoning
    "stream": False,            # Single response per prompt to reduce CPU overhead
}


# --------------------------------------------------------------------------- #
# Quantization (torchao) -- see docs/quantization.md
# --------------------------------------------------------------------------- #
# Available schemes (python run_quantize.py --list):
#   int8_weight_only                  W8A16, ~2x smaller, safest default
#   int4_weight_only                  W4A16 group-wise, ~4x smaller (sm80+)
#   int8_dynamic_act_int8_weight      W8A8, faster for large batches (sm75+)
#   float8_dynamic_act_float8_weight  FP8, needs Ada/Hopper (sm89+) -> NOT on Ampere
QUANT_SCHEME = "int8_weight_only"
QUANT_GROUP_SIZE = 128                  # Only used by int4_weight_only
QUANT_DTYPE = "bfloat16"                # Dtype for non-quantized parts / activations
QUANT_OUTPUT_ROOT = "~/nxp/models/LLMs/quantized"
QUANT_OUTPUT_DIR = os.path.join(QUANT_OUTPUT_ROOT, f"{SERVED_MODEL_NAME}-{QUANT_SCHEME}")

# Set to True to serve + evaluate the quantized checkpoint instead of the base model.
# run_pipeline.py will create the checkpoint first if it does not exist yet.
SERVE_QUANTIZED = False

if SERVE_QUANTIZED:
    BASE_MODEL_ID = MODEL_ID
    MODEL_ID = QUANT_OUTPUT_DIR
    SERVED_MODEL_NAME = f"{SERVED_MODEL_NAME}-{QUANT_SCHEME}"
    VLLM_QUANTIZATION = "torchao"       # Passed to vLLM as --quantization
    WORK_DIR = f"./eval_outputs/{SERVED_MODEL_NAME}"  # Keep results apart from the baseline
else:
    BASE_MODEL_ID = MODEL_ID
    VLLM_QUANTIZATION = ""
