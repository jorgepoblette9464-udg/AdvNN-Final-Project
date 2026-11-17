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
