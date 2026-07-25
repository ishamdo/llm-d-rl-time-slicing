# GKE End-to-End Recipe: Interleaving RL Training and Stock vLLM Batch Inference

This recipe demonstrates how to deploy **llm-d-rl-time-slicing** on Google Kubernetes Engine (GKE) to cooperatively share a single GPU node between an **RL Training Job** and an unmodified **vLLM Batch Inference Engine**.

By leveraging collaborative time-slicing mediated by the **Accelerator Orchestrator** and **Snapshot Agent**, batch inference utilizes the GPU during the RL Trainer's natural idle intervals (e.g., generation/weight evaluation steps), achieving near-100% GPU utilization without algorithmic modifications to either workload.

---

## 1. Architectural Overview

```
                          +-------------------------------------------------------------+
                          |                   GKE Time-Sliced Node                      |
                          |                 (e.g., 1x NVIDIA L4 GPU)                    |
                          |                                                             |
                          |   +-----------------------------------------------------+   |
                          |   |           DRA ResourceClaim (ExactCount=1)          |   |
                          |   +-----------------------------------------------------+   |
                          |                              ▲                              |
                          |             +----------------+----------------+             |
                          |             | Exclusive Time-Sliced Access    |             |
                          |             ▼                                 ▼             |
                          |    [ RL Trainer Pod ]               [ Shadow vLLM Pod ]     |
                          |     (Priority Job)                 (Stock vLLM Server)      |
                          |             ▲                                 ▲             |
                          +-------------|---------------------------------|-------------+
                                        | Acquire() / Yield()             | Acquire() / Dynamic Preemption
                                        ▼                                 ▼
                          +-------------------------------------------------------------+
                          |             timeslice-acceleratororchestrator               |
                          |           (Group Queue & Priority Preemption)               |
                          +-------------------------------------------------------------+
```

### Components Exposed in this Recipe:
1. **`01-trainer-gpu-claim.yaml`**: Kubernetes `resource.k8s.io/v1` Dynamic Resource Allocation (DRA) `ResourceClaim` requesting exactly one NVIDIA L4 GPU shared cooperatively across pods.
2. **`02-rl-sampler-pod.yaml`**: Dedicated 100% busy RL Sampler workload running on an isolated node (`sampler-pool`).
3. **`03-rl-trainer-pod.yaml`**: Cooperative RL Trainer workload (`rl-trainer`) that allocates persistent multi-gigabyte neural network weights in GPU HBM (`3.0 GB`), periodically acquiring the GPU for active training (`20s`) and using the Snapshot Agent (`app_endpoint` backend on port 8001) to offload real HBM memory during idle rollout phases (`120s`).
4. **`04-shadow-vllm-pod.yaml`**: Cooperative Shadow vLLM workload (`shadow-vllm`) running unmodified `Qwen/Qwen2.5-0.5B-Instruct` wrapped by a **Queue-Depth Preemption Supervisor**.
5. **`05-load-generator-pod.yaml`**: Continuous HTTP inference client (`batch-load-generator`) sending requests to `shadow-vllm-service:8000/v1/completions`.

> [!NOTE]
> **Production vs. Demo Timings:** In practice, RL sampling and training phases typically take on the order of **minutes to hours**. In this recipe, we compress these cycle times into seconds (e.g., 20s active training / 60–120s idle intervals) so you can quickly observe collaborative time-slicing and preemption in action.

---

## 2. The Queue-Depth Preemption Supervisor Pattern

A key requirement for production inference engines is **zero application modification**. Rather than modifying vLLM internals to yield cooperatively, `scripts/orchestrated_vllm_runner.py` implements a supervisor process that leverages vLLM's native **Sleep Mode** (`--enable-sleep-mode`) and the **Snapshot Agent**:

```python
with SnapshotAgentClient(AGENT_ENDPOINT) as snap_client:
    while True:
        with client.on_accelerators() as lock:
            if proc is None:
                # 1. Launch stock vLLM on first start with Sleep Mode enabled
                proc = subprocess.Popen([... "--enable-sleep-mode" ...])
            else:
                # 2. Wake up vLLM: Restore weights from CPU RAM to GPU HBM (~50-100ms!)
                try:
                    snap_client.restore_and_wait(job_id=JOB_ID, backend_config=vllm_config)
                except Exception as e:
                    # Resilient fallback: guarantee local wake_up if agent returns already-running
                    try: urllib.request.urlopen("http://127.0.0.1:8000/wake_up", data=b"")
                    except Exception: pass

            # 3. Poll Orchestrator queue depth; offload HBM if higher-priority job arrives
            while proc.poll() is None:
                time.sleep(0.5)
                status = client.get_status(group_id=GROUP_ID)
                if status.group and status.group.waiter_queue_depth > 0:
                    # 4. Offload HBM weights to CPU RAM (~1.47s) without killing the process
                    snap_client.snapshot_and_wait(job_id=JOB_ID, backend_config=vllm_config)
                    break
```

- When the RL Trainer is idle (`waiter_queue_depth == 0`), stock vLLM runs continuously serving batch completions.
- The instant the RL Trainer calls `acquire()` (`waiter_queue_depth > 0`), the supervisor calls the Snapshot Agent (`app_endpoint` backend on port 8000) to offload GPU framebuffer memory to CPU system RAM in **~1.47s** and yields the lock—without terminating the server process.
- When the RL Trainer yields (after offloading its 3.0 GB of real policy weights and optimizer states via port 8001 in **~2.44s**), the supervisor calls `restore_and_wait()` (with direct `/wake_up` fallback), copying model weights back into GPU HBM and resuming batch inference in **~50–100ms**.

---

## 3. Prerequisites & Cluster Setup

1. A GKE cluster (`v1.35+`) with Kubernetes Dynamic Resource Allocation (DRA) enabled (`NvidiaDriver` / `NvidiaDeviceClass` installed).
2. Compute prerequisites:
   - **2x NVIDIA L4 GPU nodes** across two distinct node pools:
     - `sampler-pool` (`cloud.google.com/gke-nodepool=sampler-pool`): 1x NVIDIA L4 GPU node for the dedicated RL Sampler pod.
     - `trainer-pool` (`cloud.google.com/gke-nodepool=trainer-pool`): 1x NVIDIA L4 GPU node cooperatively time-sliced between the RL Trainer and Shadow vLLM.
   - **Standard CPU node pool** (`default-pool`) for cluster services and the HTTP load generator.
3. Install the **llm-d-rl-time-slicing platform** using the turnkey installation script:

```bash
cd examples/gke-rl-batch-demo
./00-install-timeslice-platform.sh
```

This deploys:
- `timeslice-acceleratororchestrator` gRPC service (`port 50051`) in `timeslice-system`.
- `timeslice-snapshot-agent` DaemonSet on GPU nodes.
- Local binary distribution service (`timeslice-binary-server`) for in-cluster Go binaries and Python SDK bootstrap.

---

## 4. Deploying the Customer Workloads

Deploy the complete recipe into the `rl-batch-demo` namespace:

```bash
# Create the demo namespace if it does not exist
kubectl create namespace rl-batch-demo --dry-run=client -o yaml | kubectl apply -f -

# Apply ConfigMap with python scripts and workloads
kubectl create configmap rl-batch-demo-scripts --from-file=scripts/ --namespace rl-batch-demo --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f 01-trainer-gpu-claim.yaml
kubectl apply -f 02-rl-sampler-pod.yaml
kubectl apply -f 03-rl-trainer-pod.yaml
kubectl apply -f 04-shadow-vllm-pod.yaml
kubectl apply -f 05-load-generator-pod.yaml
```

Check pod status:
```bash
kubectl get pods -n rl-batch-demo
```

---

## 5. Verifying Interleaved Progress & Dynamic Preemption

### A. View Synchronized Interleaved Timeline
Run the following script to observe the 1-to-1 synchronized timeline across `rl-trainer`, `shadow-vllm`, and `batch-load-generator`:

```bash
python3 -c '
import subprocess
t_out = subprocess.check_output(["kubectl", "logs", "-n", "rl-batch-demo", "rl-trainer", "--since=180s", "--timestamps"]).decode("utf-8", errors="replace").splitlines()
v_out = subprocess.check_output(["kubectl", "logs", "-n", "rl-batch-demo", "shadow-vllm", "--since=180s", "--timestamps"]).decode("utf-8", errors="replace").splitlines()
b_out = subprocess.check_output(["kubectl", "logs", "-n", "rl-batch-demo", "batch-load-generator", "--since=180s", "--timestamps"]).decode("utf-8", errors="replace").splitlines()

events = []
for l in t_out:
    if "LOCK" in l or "Requesting" in l or "Training phase" in l:
        parts = l.split(" ", 1)
        if len(parts) == 2: events.append((parts[0], f"[RL-TRAINER]   {parts[1]}"))
for l in v_out:
    if "LOCK" in l or "Preemption signal" in l or "Requesting" in l:
        parts = l.split(" ", 1)
        if len(parts) == 2: events.append((parts[0], f"[SHADOW-VLLM]  {parts[1]}"))
for l in b_out:
    if "COMPLETED INFERENCE QUERY" in l:
        parts = l.split(" ", 1)
        if len(parts) == 2: events.append((parts[0], f"[BATCH-CLIENT] {parts[1]}"))

events.sort(key=lambda x: x[0])
for ts, msg in events[-35:]:
    print(f"{ts[:23]} | {msg}")
'
```

### Expected Output Example:
```log
2026-07-24T23:36:42.409 | [BATCH-CLIENT] COMPLETED INFERENCE QUERY #1: "In a distributed computing system, it is often necessary to perform tasks i..."
2026-07-24T23:36:42.589 | [BATCH-CLIENT] COMPLETED INFERENCE QUERY #2: "To understand the advantages of cooperative accelerator time-slicing, let's..."
2026-07-24T23:36:42.774 | [BATCH-CLIENT] COMPLETED INFERENCE QUERY #3: "Cooperative accelerator time-slicing is a technique used in distributed com..."
2026-07-24T23:36:42.850 | [BATCH-CLIENT] COMPLETED INFERENCE QUERY #4: "To provide a comprehensive answer, I will discuss the..."
2026-07-24T23:36:44.120 | [SHADOW-VLLM]  [Shadow vLLM] Preemption signal detected (waiter_queue_depth=1). Offloading HBM memory via Snapshot Agent...
2026-07-24T23:36:44.331 | [SHADOW-VLLM]  [Shadow vLLM] HBM offloaded to CPU memory in 1472 ms. Yielding lock...
2026-07-24T23:36:44.350 | [SHADOW-VLLM]  [Shadow vLLM] LOCK YIELDED TO HIGHER-PRIORITY TRAINER.
2026-07-24T23:36:45.102 | [RL-TRAINER]   [RL Trainer Iter 1] LOCK ACQUIRED (Waited 1001 ms).
2026-07-24T23:36:45.205 | [RL-TRAINER]   [RL Trainer] Initializing Realistic RL Policy Model (~3.0 GB target VRAM footprint including gradients/optimizer)...
2026-07-24T23:36:45.310 | [RL-TRAINER]   [RL Trainer Iter 1] Executing active GPU policy optimization (20s)...
2026-07-24T23:37:05.312 | [RL-TRAINER]   [RL Trainer Iter 1] Training phase completed. Offloading HBM memory via Snapshot Agent before yielding...
2026-07-24T23:37:07.754 | [RL-TRAINER]   [RL Trainer HTTP] Model & optimizer offloaded to CPU in 2442 ms.
2026-07-24T23:37:07.780 | [RL-TRAINER]   [RL Trainer Iter 1] LOCK YIELDED. Entering idle CPU rollout/eval sleep (120s)...
2026-07-24T23:37:08.120 | [SHADOW-VLLM]  [Shadow vLLM] LOCK ACQUIRED (Waited 23780 ms). Restoring HBM memory via Snapshot Agent / local wake_up (~50-100ms resume)...
2026-07-24T23:37:08.215 | [SHADOW-VLLM]  [Shadow vLLM] VRAM restored from CPU host RAM. Resuming batch inference.
2026-07-24T23:37:09.110 | [BATCH-CLIENT] COMPLETED INFERENCE QUERY #5: "Cooperative acceleration time-slicing allows distributed systems to share..."
```

> [!NOTE]
> **Realistic POC Enhancements Reflected in this Demo:**
> - **Realistic Trainer Weights (`3.0 GB`):** Instead of an empty loop, `rl_trainer.py` allocates multi-layer neural network weights and AdamW optimizer states in HBM, sized to prevent node cgroup OOM kills on standard 16 GB GKE VMs during CPU memory offloading.
> - **Application-Aware Offloading over DRA:** Bypasses process-level `cuda-checkpoint` driver limitations in Kubernetes DRA environments by using native HTTP endpoints (`app_endpoint` backend on port 8000 for vLLM Sleep Mode and port 8001 for RL Trainer).
> - **Local Disk Model Caching (`hostPath`):** Mounts `/tmp/huggingface_cache` -> `/root/.cache` in vLLM pods for instant model loading across restarts without network download delays.
> - **Resilient Wake-Up Fallbacks:** Guarantees instant ~50–100 ms GPU resumption via local HTTP `/wake_up` fallbacks even if NVML activity detection marks a pod as `RUNNING` prior to gRPC restore completion.

## 6. Cleanup

Remove the demo namespace and platform components when finished:

```bash
# 1. Remove the customer workloads namespace
kubectl delete namespace rl-batch-demo --ignore-not-found

# 2. (Optional) Uninstall the llm-d-rl-time-slicing platform
helm uninstall timeslice -n timeslice-system
kubectl delete namespace timeslice-system --ignore-not-found
```
