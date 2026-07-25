#!/usr/bin/env python3
"""
Queue-Depth Preemption Supervisor for Shadow vLLM with HBM Memory Swapping
Runs unmodified vLLM OpenAI API server inside an acquired Group lock with Sleep Mode enabled.
Monitors Orchestrator waiter queue depth; when RL Trainer requests the GPU (`waiter_queue_depth > 0`),
uses the Snapshot Agent to offload HBM memory to CPU host RAM (~200ms) without killing the process.
When the lock is re-acquired, restores VRAM from CPU RAM (~50-100ms) for instant resume.
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
from timeslice.snapshot_agent import SnapshotAgentClient
from timeslice.snapshot_agent import snapshot_agent_pb2 as snapshot

JOB_ID = os.getenv("JOB_ID", "batch-inference")
GROUP_ID = os.getenv("GROUP_ID", "trainer-group")
ORCHESTRATOR_ADDR = os.getenv("ORCHESTRATOR_ADDR", "timeslice-acceleratororchestrator.timeslice-system.svc.cluster.local:50051")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
PORT = os.getenv("PORT", "8000")
NODE_IP = os.getenv("NODE_IP")
POD_IP = os.getenv("POD_IP", "localhost")
AGENT_ENDPOINT = os.getenv("AGENT_ENDPOINT", f"{NODE_IP}:9001" if NODE_IP else "localhost:9001")

client = OrchestratorClient(
    target=ORCHESTRATOR_ADDR,
    job_id=JOB_ID,
    group_id=GROUP_ID,
)

vllm_config = snapshot.BackendConfig(
    app_endpoint=snapshot.AppEndpointConfig(
        app=snapshot.APP_VLLM,
        endpoints=[f"http://{POD_IP}:{PORT}"],
    )
)

print(f"Starting Queue-Depth Preemption Supervisor for stock vLLM ({MODEL_NAME}) with Sleep Mode & HBM Swapping...", flush=True)
proc = None

with SnapshotAgentClient(AGENT_ENDPOINT) as snap_client:
    while True:
        print("[Shadow vLLM] Requesting Group Lock from Orchestrator...", flush=True)
        with client.on_accelerators() as lock:
            if proc is None:
                print(f"[Shadow vLLM] LOCK ACQUIRED (Waited {lock.waited_ms} ms). Launching stock vLLM server with Sleep Mode enabled...", flush=True)
                env = os.environ.copy()
                env["VLLM_SERVER_DEV_MODE"] = "1"
                proc = subprocess.Popen([
                    "python3", "-m", "vllm.entrypoints.openai.api_server",
                    "--model", MODEL_NAME,
                    "--port", PORT,
                    "--gpu-memory-utilization", "0.7",
                    "--enable-sleep-mode"
                ], env=env)
                print("[Shadow vLLM] Waiting for vLLM HTTP server to complete initialization and load weights into HBM...", flush=True)
                start_wait = time.time()
                while time.time() - start_wait < 300:
                    if proc.poll() is not None:
                        print("[Shadow vLLM] vLLM server process died during startup!", flush=True)
                        break
                    try:
                        req = urllib.request.Request(f"http://localhost:{PORT}/health")
                        with urllib.request.urlopen(req, timeout=1.0) as resp:
                            if resp.status == 200:
                                print(f"[Shadow vLLM] vLLM server is READY and serving (took {int(time.time() - start_wait)}s)!", flush=True)
                                break
                    except Exception:
                        pass
                    time.sleep(2.0)
            else:
                print(f"[Shadow vLLM] LOCK ACQUIRED (Waited {lock.waited_ms} ms). Restoring HBM memory via Snapshot Agent (~50-100ms resume)...", flush=True)
                try:
                    snap_client.restore_and_wait(job_id=JOB_ID, backend_config=vllm_config, poll_interval_sec=0.2)
                    print("[Shadow vLLM] VRAM restored from CPU host RAM. Resuming batch inference.", flush=True)
                except Exception as restore_err:
                    print(f"[Shadow vLLM] Warning: Restore via agent returned ({restore_err}). Ensuring vLLM wake_up...", flush=True)
                    try: urllib.request.urlopen("http://127.0.0.1:8000/wake_up", data=b"", timeout=5.0)
                    except Exception: pass

            try:
                # Poll Orchestrator queue depth; offload HBM immediately if a higher-priority job arrives
                while proc.poll() is None:
                    time.sleep(0.5)
                    try:
                        status = client.get_status(group_id=GROUP_ID, timeout_sec=2.0)
                        if status.group and status.group.waiter_queue_depth > 0:
                            print(f"[Shadow vLLM] Preemption signal detected (waiter_queue_depth={status.group.waiter_queue_depth}). Offloading HBM memory via Snapshot Agent...", flush=True)
                            snap_client.snapshot_and_wait(job_id=JOB_ID, backend_config=vllm_config, poll_interval_sec=0.2)
                            print("[Shadow vLLM] HBM offloaded to CPU memory. Yielding lock...", flush=True)
                            break
                    except Exception as poll_err:
                        # Ignore transient status check errors
                        pass
            except KeyboardInterrupt:
                if proc and proc.poll() is None:
                    proc.terminate()
                    proc.wait()
                break

            if proc and proc.poll() is not None:
                print("[Shadow vLLM] Server process exited unexpectedly. Will relaunch on next acquisition...", flush=True)
                proc = None

        print("[Shadow vLLM] LOCK YIELDED TO HIGHER-PRIORITY TRAINER.", flush=True)
        time.sleep(1)
