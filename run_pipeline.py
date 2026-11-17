import subprocess
import time
import urllib.request
import urllib.error
import sys
import signal
import config
import run_eval


def wait_for_server(url: str, timeout: int = 300) -> bool:
    """Polls the vLLM server until it is ready to accept requests."""
    health_url = f"{url}/models"
    start_time = time.time()
    print(f"[ORCHESTRATOR] Waiting for vLLM server at {health_url}...")

    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    print("[ORCHESTRATOR] Server is online and ready!")
                    return True
        except (urllib.error.URLError, ConnectionResetError):
            time.sleep(3)
    return False


def main():
    # Launch vLLM in its own process group so we can cleanly kill child workers on exit
    server_proc = subprocess.Popen(
        ["bash", "./start_server.sh"],
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
    )

    try:
        if not wait_for_server(config.API_URL):
            print("[ERROR] vLLM server failed to start within timeout.", file=sys.stderr)
            sys.exit(1)

        # Run the EvalScope benchmark
        run_eval.main()

    except KeyboardInterrupt:
        print("\n[ORCHESTRATOR] Interrupted by user. Partial progress has been saved.")
    finally:
        print("[ORCHESTRATOR] Shutting down vLLM server and releasing GPU memory...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("[ORCHESTRATOR] Cleanup complete.")


if __name__ == "__main__":
    main()
