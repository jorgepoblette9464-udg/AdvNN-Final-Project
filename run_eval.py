import os
# Cap CPU threads for the evaluation client process as well
import config
os.environ["OMP_NUM_THREADS"] = config.CPU_THREADS
os.environ["MKL_NUM_THREADS"] = config.CPU_THREADS

from evalscope import TaskConfig, run_task


def get_task_config() -> TaskConfig:
    # Check if a previous run directory exists to enable automatic resumption
    resume_dir = config.WORK_DIR if os.path.exists(config.WORK_DIR) else None

    if resume_dir:
        print(f"[INFO] Found existing output directory '{resume_dir}'. Resuming from last checkpoint...")
    else:
        print(f"[INFO] Starting fresh evaluation in '{config.WORK_DIR}'...")

    return TaskConfig(
        model=config.SERVED_MODEL_NAME,
        api_url=config.API_URL,
        api_key="EMPTY",
        eval_type="openai_api",          # Connects to the local vLLM server
        datasets=config.DATASETS,        # ['mmlu', 'mmlu_pro', 'hellaswag']
        eval_batch_size=config.EVAL_BATCH_SIZE,
        work_dir=config.WORK_DIR,        # Pin output directory instead of timestamped folders
        use_cache=resume_dir,            # Skips already-completed samples if interrupted
        generation_config={
            "temperature": 0.0,          # Greedy decoding for deterministic, reproducible benchmarks
            "max_tokens": 512,           # Sufficient for MMLU-Pro chain-of-thought/answers
        },
        # Optional: Uncomment limit to test your pipeline on 10 samples per subset first
        # limit=10,
    )


def main():
    task_cfg = get_task_config()
    results = run_task(task_cfg=task_cfg)
    print("\n[SUCCESS] Evaluation Complete. Final Summary:")
    print(results)


if __name__ == "__main__":
    main()
