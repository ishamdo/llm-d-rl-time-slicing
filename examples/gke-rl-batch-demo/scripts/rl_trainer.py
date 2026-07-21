#!/usr/bin/env python3
"""
Simulated RL Trainer Script with Time-Slicing
Acquires Group lock during 20s active training phase and yields lock during 20s idle/sleep phase.
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

import torch
from timeslice import OrchestratorClient

JOB_ID = os.getenv("JOB_ID", "rl-trainer")
GROUP_ID = os.getenv("GROUP_ID", "trainer-group")
ORCHESTRATOR_ADDR = os.getenv("ORCHESTRATOR_ADDR", "timeslice-acceleratororchestrator.timeslice-system.svc.cluster.local:50051")

BUSY_SECONDS = int(os.getenv("BUSY_SECONDS", "20"))
IDLE_SECONDS = int(os.getenv("IDLE_SECONDS", "60"))

client = OrchestratorClient(
    target=ORCHESTRATOR_ADDR,
    job_id=JOB_ID,
    group_id=GROUP_ID,
)

def gpu_busy_work(duration_sec: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    end_time = time.time() + duration_sec
    while time.time() < end_time:
        _ = torch.matmul(a, b)

print(f"Starting Simulated RL Trainer Loop (Busy={BUSY_SECONDS}s, Idle={IDLE_SECONDS}s)...", flush=True)
iteration = 0
while True:
    iteration += 1
    print(f"[RL Trainer Iter {iteration}] Requesting accelerator lock from Orchestrator...", flush=True)
    with client.on_accelerators() as lock:
        print(f"[RL Trainer Iter {iteration}] LOCK ACQUIRED (Waited {lock.waited_ms} ms). Executing GPU Training ({BUSY_SECONDS}s)...", flush=True)
        gpu_busy_work(BUSY_SECONDS)
        print(f"[RL Trainer Iter {iteration}] Training phase completed. Yielding lock...", flush=True)
    
    print(f"[RL Trainer Iter {iteration}] LOCK YIELDED. Entering idle sleep ({IDLE_SECONDS}s)...", flush=True)
    time.sleep(IDLE_SECONDS)
