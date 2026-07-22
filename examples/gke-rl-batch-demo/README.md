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
3. **`03-rl-trainer-pod.yaml`**: Cooperative RL Trainer workload (`rl-trainer`) that periodically acquires the GPU for active training (`20s`) and yields during idle phases (`60–120s`).
4. **`04-shadow-vllm-pod.yaml`**: Cooperative Shadow vLLM workload (`shadow-vllm`) running unmodified `Qwen/Qwen2.5-0.5B-Instruct` wrapped by a **Queue-Depth Preemption Supervisor**.
5. **`05-load-generator-pod.yaml`**: Continuous HTTP inference client (`batch-load-generator`) sending requests to `shadow-vllm-service:8000/v1/completions`.

> [!NOTE]
> **Production vs. Demo Timings:** In practice, RL sampling and training phases typically take on the order of **minutes to hours**. In this recipe, we compress these cycle times into seconds (e.g., 20s active training / 60–120s idle intervals) so you can quickly observe collaborative time-slicing and preemption in action.

---

## 2. The Queue-Depth Preemption Supervisor Pattern

A key requirement for production inference engines is **zero application modification**. Rather than modifying vLLM internals to yield cooperatively, `scripts/orchestrated_vllm_runner.py` implements a supervisor process:

```python
with client.on_accelerators() as lock:
    # 1. Launch stock unmodified vLLM OpenAI API server
    proc = subprocess.Popen([
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--port", "8000",
        "--gpu-memory-utilization", "0.7"
    ])
    try:
        # 2. Poll Orchestrator group status every 1s
        while proc.poll() is None:
            time.sleep(1.0)
            status = client.get_status(group_id=GROUP_ID, timeout_sec=2.0)
            # 3. Preempt immediately when higher-priority RL Trainer requests access
            if status.group and status.group.waiter_queue_depth > 0:
                proc.terminate()
                proc.wait(timeout=5)
                break
    except KeyboardInterrupt:
        proc.terminate()
```

- When the RL Trainer is idle (`waiter_queue_depth == 0`), stock vLLM runs continuously serving batch completions.
- The instant the RL Trainer calls `acquire()` (`waiter_queue_depth > 0`), the supervisor cleanly terminates stock vLLM (`SIGTERM`), exits the context manager, and hands over GPU memory.

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
2026-07-21T02:52:44.126 | [BATCH-CLIENT] COMPLETED INFERENCE QUERY #1: "The advantage of cooperative accelerator time-slicing is that it allows mul..."
2026-07-21T02:52:45.041 | [BATCH-CLIENT] COMPLETED INFERENCE QUERY #6: "Cooperative acceleration time-slicing (CATS) is a technique used in compute..."
2026-07-21T02:52:47.758 | [BATCH-CLIENT] COMPLETED INFERENCE QUERY #21: "In a cooperative accelerator, multiple processors are assigned to different..."
2026-07-21T02:52:57.782 | [BATCH-CLIENT] COMPLETED INFERENCE QUERY #20: "Cooperative accelerator time-slicing is a technique used in distributed com..."
2026-07-21T02:52:57.835 | [RL-TRAINER]   [RL Trainer Iter 3] Requesting accelerator lock from Orchestrator...
2026-07-21T02:52:57.983 | [SHADOW-VLLM]  [Shadow vLLM] Preemption signal detected (waiter_queue_depth=1). Terminating stock vLLM...
2026-07-21T02:53:04.017 | [SHADOW-VLLM]  [Shadow vLLM] LOCK YIELDED TO HIGHER-PRIORITY TRAINER.
2026-07-21T02:53:04.842 | [RL-TRAINER]   [RL Trainer Iter 3] LOCK ACQUIRED (Waited 7001 ms). Executing GPU Training (20s)...
2026-07-21T02:53:24.855 | [RL-TRAINER]   [RL Trainer Iter 3] Training phase completed. Yielding lock...
2026-07-21T02:53:24.888 | [RL-TRAINER]   [RL Trainer Iter 3] LOCK YIELDED. Entering idle sleep (120s)...
2026-07-21T02:53:25.022 | [SHADOW-VLLM]  [Shadow vLLM] LOCK ACQUIRED (Waited 20000 ms). Launching stock vLLM server...
```

---

## 6. Cleanup

Remove the demo namespace and platform components when finished:

```bash
# 1. Remove the customer workloads namespace
kubectl delete namespace rl-batch-demo --ignore-not-found

# 2. (Optional) Uninstall the llm-d-rl-time-slicing platform
helm uninstall timeslice -n timeslice-system
kubectl delete namespace timeslice-system --ignore-not-found
```
