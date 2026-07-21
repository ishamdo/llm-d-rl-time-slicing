#!/usr/bin/env python3
"""
Queue-Depth Preemption Supervisor for Shadow vLLM
Runs unmodified vLLM OpenAI API server inside an acquired Group lock.
Monitors Orchestrator waiter queue depth; when RL Trainer requests the GPU (`waiter_queue_depth > 0`),
preempts vLLM immediately and yields the lock.
"""
import os
import sys
import subprocess
import time
import urllib.request
import tarfile

# Bootstrap timeslice SDK and gRPC dependencies
print("Bootstrapping timeslice Python SDK from in-cluster binary server...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "grpcio>=1.64.1", "protobuf>=5.29.0"])

SDK_DIR = "/tmp/sdk"
if not os.path.exists(os.path.join(SDK_DIR, "timeslice")):
    os.makedirs(SDK_DIR, exist_ok=True)
    urllib.request.urlretrieve(
        "http://timeslice-binary-server.timeslice-system.svc.cluster.local:8080/timeslice-sdk.tar.gz",
        os.path.join(SDK_DIR, "sdk.tar.gz")
    )
    with tarfile.open(os.path.join(SDK_DIR, "sdk.tar.gz"), "r:gz") as tar:
        tar.extractall(SDK_DIR)
sys.path.insert(0, SDK_DIR)

from timeslice import OrchestratorClient

JOB_ID = os.getenv("JOB_ID", "batch-inference")
GROUP_ID = os.getenv("GROUP_ID", "trainer-group")
ORCHESTRATOR_ADDR = os.getenv("ORCHESTRATOR_ADDR", "timeslice-acceleratororchestrator.timeslice-system.svc.cluster.local:50051")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
PORT = os.getenv("PORT", "8000")

client = OrchestratorClient(
    target=ORCHESTRATOR_ADDR,
    job_id=JOB_ID,
    group_id=GROUP_ID,
)

print(f"Starting Queue-Depth Preemption Supervisor for stock vLLM ({MODEL_NAME})...", flush=True)
while True:
    print("[Shadow vLLM] Requesting Group Lock from Orchestrator...", flush=True)
    with client.on_accelerators() as lock:
        print(f"[Shadow vLLM] LOCK ACQUIRED (Waited {lock.waited_ms} ms). Launching stock vLLM server...", flush=True)
        proc = subprocess.Popen([
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_NAME,
            "--port", PORT,
            "--gpu-memory-utilization", "0.7"
        ])
        try:
            # Poll Orchestrator queue depth; preempt immediately if a higher-priority job arrives
            while proc.poll() is None:
                time.sleep(1.0)
                try:
                    status = client.get_status(group_id=GROUP_ID, timeout_sec=2.0)
                    if status.group and status.group.waiter_queue_depth > 0:
                        print(f"[Shadow vLLM] Preemption signal detected (waiter_queue_depth={status.group.waiter_queue_depth}). Terminating stock vLLM...", flush=True)
                        proc.terminate()
                        proc.wait(timeout=5)
                        break
                except Exception as poll_err:
                    # Ignore transient status check errors
                    pass
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
    print("[Shadow vLLM] LOCK YIELDED TO HIGHER-PRIORITY TRAINER.", flush=True)
    time.sleep(1)
