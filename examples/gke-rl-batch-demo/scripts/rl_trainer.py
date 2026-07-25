#!/usr/bin/env python3
"""
Simulated RL Trainer Script with Real Model Weights & Time-Slicing
Loads persistent multi-gigabyte neural network weights into GPU HBM.
Acquires Group lock during active training phase (`BUSY_SECONDS`), executes gradient updates,
and uses the Snapshot Agent (`cuda` backend) to swap real HBM memory out to host CPU RAM
before yielding the lock for the idle rollout valley (`IDLE_SECONDS`).
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
import torch.nn as nn
from timeslice import OrchestratorClient
from timeslice.snapshot_agent import SnapshotAgentClient
from timeslice.snapshot_agent import snapshot_agent_pb2 as snapshot

JOB_ID = os.getenv("JOB_ID", "rl-trainer")
GROUP_ID = os.getenv("GROUP_ID", "trainer-group")
ORCHESTRATOR_ADDR = os.getenv("ORCHESTRATOR_ADDR", "timeslice-acceleratororchestrator.timeslice-system.svc.cluster.local:50051")
NODE_IP = os.getenv("NODE_IP")
AGENT_ENDPOINT = os.getenv("AGENT_ENDPOINT", f"{NODE_IP}:9001" if NODE_IP else "localhost:9001")

BUSY_SECONDS = int(os.getenv("BUSY_SECONDS", "20"))
IDLE_SECONDS = int(os.getenv("IDLE_SECONDS", "120"))
MODEL_VRAM_GB = float(os.getenv("MODEL_VRAM_GB", "3.0"))

client = OrchestratorClient(
    target=ORCHESTRATOR_ADDR,
    job_id=JOB_ID,
    group_id=GROUP_ID,
)

POD_IP = os.getenv("POD_IP", "localhost")
PORT = int(os.getenv("PORT", "8001"))

app_config = snapshot.BackendConfig(
    app_endpoint=snapshot.AppEndpointConfig(
        app=snapshot.APP_VLLM,
        endpoints=[f"http://{POD_IP}:{PORT}"],
    )
)

from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

class OffloadHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global model, optimizer, device
        if "/sleep" in self.path:
            t0 = time.time()
            if model is not None:
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                model.to("cpu")
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to("cpu", non_blocking=True)
                torch.cuda.empty_cache()
            ms = int((time.time() - t0) * 1000)
            print(f"[RL Trainer HTTP] Model & optimizer offloaded to CPU in {ms} ms.", flush=True)
            self.send_response(200)
            self.end_headers()
        elif "/wake_up" in self.path or "/wake" in self.path:
            t0 = time.time()
            if model is not None and device is not None:
                model.to(device)
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device, non_blocking=True)
            ms = int((time.time() - t0) * 1000)
            print(f"[RL Trainer HTTP] Model & optimizer restored to GPU in {ms} ms.", flush=True)
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def start_http_server(port):
    server = HTTPServer(("0.0.0.0", port), OffloadHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[RL Trainer] Application-aware offload HTTP server listening on port {port}...", flush=True)

start_http_server(PORT)

class RealisticRLModel(nn.Module):
    """A realistic deep Transformer-like model layer structure designed to occupy target VRAM."""
    def __init__(self, target_vram_gb: float):
        super().__init__()
        # Each float32 is 4 bytes. For target_vram_gb, total elements = (target_vram_gb * 1024^3) / 4
        # We split this across multiple linear layers to simulate realistic weight matrices and gradients.
        total_bytes = int(target_vram_gb * (1024 ** 3))
        # Account for optimizer state (Adam requires 2x weight memory for momentum/variance + gradients)
        # So weights should be roughly 1/4th of total target VRAM to reach target footprint during training.
        weight_bytes = total_bytes // 4
        dim = int((weight_bytes / 4 / 10) ** 0.5)  # 10 layers
        dim = max(dim, 1024)
        print(f"[RL Trainer] Initializing Realistic RL Policy Model with hidden dim={dim} (~{target_vram_gb} GB target VRAM footprint including gradients/optimizer)...", flush=True)
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            ) for _ in range(5)
        ])
        self.dim = dim

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

def create_model_and_optimizer(device):
    model = RealisticRLModel(MODEL_VRAM_GB).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return model, optimizer

def gpu_busy_work(model, optimizer, device, duration_sec: int):
    model.train()
    end_time = time.time() + duration_sec
    step = 0
    while time.time() < end_time:
        step += 1
        x = torch.randn(32, model.dim, device=device)
        optimizer.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"  [RL Training Step {step}] Loss: {loss.item():.4f} | VRAM Allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB", flush=True)
        time.sleep(0.01) # Small sleep to simulate compute pacing without freezing stdout

print(f"Starting Realistic RL Trainer Loop (Busy={BUSY_SECONDS}s, Idle={IDLE_SECONDS}s, Target VRAM={MODEL_VRAM_GB}GB)...", flush=True)
model, optimizer, device = None, None, None

with SnapshotAgentClient(AGENT_ENDPOINT) as snap_client:
    iteration = 0
    while True:
        iteration += 1
        print(f"[RL Trainer Iter {iteration}] Requesting accelerator lock from Orchestrator...", flush=True)
        with client.on_accelerators() as lock:
            print(f"[RL Trainer Iter {iteration}] LOCK ACQUIRED (Waited {lock.waited_ms} ms).", flush=True)
            
            if model is None:
                print("[RL Trainer] First acquisition: Allocating model weights and optimizer in GPU HBM...", flush=True)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model, optimizer = create_model_and_optimizer(device)
            else:
                print("[RL Trainer] Restoring model HBM memory via Snapshot Agent (app_endpoint backend)...", flush=True)
                t0 = time.time()
                try:
                    res = snap_client.restore_and_wait(job_id=JOB_ID, backend_config=app_config, poll_interval_sec=0.2)
                    restore_ms = int((time.time() - t0) * 1000)
                    print(f"[RL Trainer] VRAM successfully restored from CPU host RAM in {restore_ms} ms.", flush=True)
                except Exception as restore_err:
                    print(f"[RL Trainer] Warning: Snapshot restore via agent returned ({restore_err}). Ensuring local wake_up...", flush=True)
                    try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/wake_up", data=b"", timeout=5.0)
                    except Exception: pass

            print(f"[RL Trainer Iter {iteration}] Executing active GPU policy optimization ({BUSY_SECONDS}s)...", flush=True)
            gpu_busy_work(model, optimizer, device, BUSY_SECONDS)
            
            print(f"[RL Trainer Iter {iteration}] Training phase completed. Offloading HBM memory via Snapshot Agent before yielding...", flush=True)
            t0 = time.time()
            try:
                res = snap_client.snapshot_and_wait(job_id=JOB_ID, backend_config=app_config, poll_interval_sec=0.2)
                snap_ms = int((time.time() - t0) * 1000)
                print(f"[RL Trainer] Model HBM weights offloaded to CPU system RAM in {snap_ms} ms.", flush=True)
            except Exception as snap_err:
                print(f"[RL Trainer] Warning: Snapshot offload failed ({snap_err}).", flush=True)

        print(f"[RL Trainer Iter {iteration}] LOCK YIELDED. Entering idle CPU rollout/eval sleep ({IDLE_SECONDS}s)...", flush=True)
        time.sleep(IDLE_SECONDS)
