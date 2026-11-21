#!/usr/bin/env bash
set -e

# Read variables from config.py to keep a single source of truth
eval $(python3 -c "
import config
print(f'export CUDA_VISIBLE_DEVICES={config.GPU_ID}')
print(f'export OMP_NUM_THREADS={config.CPU_THREADS}')
print(f'export MKL_NUM_THREADS={config.CPU_THREADS}')
print(f'export VLLM_USE_MODELSCOPE=True')
print(f'MODEL_ID={config.MODEL_ID}')
print(f'SERVED_NAME={config.SERVED_MODEL_NAME}')
print(f'HOST={config.HOST}')
print(f'PORT={config.PORT}')
print(f'GPU_UTIL={config.GPU_MEM_UTILIZATION}')
print(f'MAX_LEN={config.MAX_MODEL_LEN}')
print(f'QUANT_METHOD={config.VLLM_QUANTIZATION}')
")

# Extra vLLM args, e.g. --quantization torchao when SERVE_QUANTIZED=True in config.py
EXTRA_ARGS=()
if [ -n "$QUANT_METHOD" ]; then
    echo "Serving quantized checkpoint ($QUANT_METHOD): $MODEL_ID"
    EXTRA_ARGS+=(--quantization "$QUANT_METHOD")
fi

echo "Starting vLLM on GPU $CUDA_VISIBLE_DEVICES with $GPU_UTIL VRAM cap and $OMP_NUM_THREADS CPU threads..."
export VLLM_USE_V1=0


# Disable FlashInfer JIT compilation (no nvcc available on shared server)
export VLLM_FLASHINFER_FORCE_TENSOR_CORES=0
export VLLM_USE_FLASHINFER_SAMPLER=0

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --enforce-eager \
    --enable-prefix-caching \
    --served-model-name "$SERVED_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --dtype auto \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}"
    
