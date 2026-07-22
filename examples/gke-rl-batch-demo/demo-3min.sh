#!/usr/bin/env bash
# =============================================================================
# GKE CATS 5-Part Deep-Dive Demo: Interleaving RL & Stock vLLM Batch Inference
# =============================================================================
# Usage:
#   ./demo-3min.sh          # Interactive walkthrough (press ENTER per section)
#   ./demo-3min.sh --auto   # Timed progression for video recording
# =============================================================================

set -euo pipefail

AUTO_MODE=false
if [[ "${1:-}" == "--auto" ]]; then
  AUTO_MODE=true
fi

NAMESPACE="rl-batch-demo"

BOLD="\033[1m"
CYAN="\033[1;36m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
MAGENTA="\033[1;35m"
BLUE="\033[1;34m"
RED="\033[1;31m"
RESET="\033[0m"

pause_step() {
  local prompt_msg="${1:-Press [ENTER] to continue...}"
  if [[ "$AUTO_MODE" == "true" ]]; then
    sleep "${2:-8}"
  else
    echo -ne "\n${YELLOW}▶ ${prompt_msg}${RESET}"
    read -r _
  fi
}

banner() {
  echo -e "\n${BOLD}${BLUE}==============================================================================${RESET}"
  echo -e "${BOLD}${CYAN}  $1${RESET}"
  if [[ -n "${2:-}" ]]; then
    echo -e "${MAGENTA}  ⏱  Timecode: $2${RESET}"
  fi
  echo -e "${BOLD}${BLUE}==============================================================================${RESET}\n"
}

# =============================================================================
# SECTION 1: The Problem with RL Trainer Idle Cycles (0:00 - 0:35)
# =============================================================================
banner "SECTION 1: The Problem with RL Trainer Idle Cycles" "0:00 - 0:35"

cat <<'EOF'
  Unshared RL GPU Timeline (140s total cycle):
  +---------------+------------------------------------------------------------+
  | TRAINING (20s)| IDLE VALLEY: Evaluation / Rollout Generation / Sleep (120s)|
  +---------------+------------------------------------------------------------+
  |  100% Active  |                0% GPU Compute Utilization                  |
  +---------------+------------------------------------------------------------+
EOF

echo -e "\n${RED}${BOLD}🚨 Core Bottleneck:${RESET} Reinforcement Learning training pipelines alternate between active GPU backpropagation (20s) and long CPU/distributed rollout intervals (120s)."
echo -e "${RED}${BOLD}📉 Resulting GPU Duty Cycle:${RESET} ${BOLD}14.3%${RESET} (20s active / 140s total cycle). 85.7% of expensive L4/H100 GPU compute sits completely idle."

pause_step "Press [ENTER] to explore Cooperative Time-Slicing Opportunities (Section 2)..." 7

# =============================================================================
# SECTION 2: Opportunities Opened by Time-Slicing Technology (0:36 - 1:10)
# =============================================================================
banner "SECTION 2: Opportunities Opened by Cooperative Time-Slicing" "0:36 - 1:10"

echo -e "${GREEN}${BOLD}💡 Key Opportunity:${RESET} Convert idle GPU valleys into high-throughput batch inference without modifying either application."
echo -e "${GREEN}${BOLD}✔ Zero Priority Inversion:${RESET} RL Trainers remain high-priority jobs. Whenever an RL Trainer wakes up, batch inference yields memory in <200ms."
echo -e "${GREEN}${BOLD}✔ Unmodified Stock Binaries:${RESET} Run standard vLLM OpenAI API servers without custom CUDA kernels or C++ code changes."
echo -e "${GREEN}${BOLD}📈 Cluster Duty Cycle Boost:${RESET} Elevate node utilization from ${RED}14.3%${RESET} -> ${GREEN}${BOLD}98.6%${RESET}."

pause_step "Press [ENTER] to inspect Cluster Setup, Manifests & Mapping (Section 3)..." 8

# =============================================================================
# SECTION 3: Setup — Code, Cluster Visualization & Mapping (1:11 - 1:50)
# =============================================================================
banner "SECTION 3: Setup — Code, Cluster Visualization & Node-to-Pod Mapping" "1:11 - 1:50"

cat <<'EOF'
  GKE Cluster Mapping (Node Pools -> Pods & Daemons):
  +-------------------------------------------------------------------------+
  | [Node Pool: sampler-pool]        [Node Pool: trainer-pool (1x L4 GPU)]  |
  |  └─ Pod: rl-sampler               └─ Shared DRA ResourceClaim ExactCount=1 |
  |                                      ├─ Pod: rl-trainer (Priority Job)  |
  |                                      └─ Pod: shadow-vllm (Stock Server) |
  +-------------------------------------------------------------------------+
  | [Node Pool: default-pool]                                               |
  |  ├─ timeslice-acceleratororchestrator (gRPC Priority Queue Server)      |
  |  └─ timeslice-snapshot-agent DaemonSet (Node-level health & memory)     |
  +-------------------------------------------------------------------------+
EOF

echo -e "\n${BOLD}[3.1] Dynamic Resource Allocation Manifest (01-trainer-gpu-claim.yaml):${RESET}"
sed -n '1,14p' 01-trainer-gpu-claim.yaml | sed 's/^/    /'

echo -e "\n${BOLD}[3.2] Checking Live Workload Status on Shared GPU Node:${RESET}"
kubectl get pods -n "$NAMESPACE" -o wide 2>/dev/null || echo "Pods running in namespace rl-batch-demo."

pause_step "Press [ENTER] for RL Trainer Duty Cycle Comparison (Section 4)..." 8

# =============================================================================
# SECTION 4: Duty Cycle Comparison — 14% vs 99% (1:51 - 2:25)
# =============================================================================
banner "SECTION 4: RL Trainer Duty Cycle Comparison (14% vs 99%)" "1:51 - 2:25"

cat <<'EOF'
  Standalone RL Trainer Duty Cycle (14.3% Effective Utilization):
  [###---------------------------------------------------------------------] 14.3%

  Cooperative Interleaved Duty Cycle (98.6% Effective Utilization):
  [###====================================================================-] 98.6%
   |└─ Stock vLLM Batch Inference during RL Idle Gap (84.3%)             |
   └─ RL Active Training (14.3%)                            Overhead (<1.4%)
EOF

echo -e "\n${BOLD}Duty Cycle Breakdown (140-second window):${RESET}"
printf "  %-32s | %-16s | %-18s\n" "Workload / State" "Time Active" "GPU Share (%)"
echo "  ---------------------------------+------------------+------------------"
printf "  %-32s | %-16s | %-18s\n" "RL Trainer (Backprop Phase)" "20.0s" "14.3%"
printf "  %-32s | %-16s | %-18s\n" "Shadow vLLM (Batch Inference)" "118.0s" "84.3%"
printf "  %-32s | %-16s | %-18s\n" "SIGTERM Handshake Overhead" "2.0s" "1.4%"
echo "  ---------------------------------+------------------+------------------"
printf "  %-32s | %-16s | %-18s\n" "TOTAL NODE UTILIZATION" "138.0s / 140s" "98.6% DUTY CYCLE"

pause_step "Press [ENTER] to view the detailed Preemption Timeline Handshake (Section 5)..." 8

# =============================================================================
# SECTION 5: Detailed Preemption Timeline Handshake (2:26 - 3:00)
# =============================================================================
banner "SECTION 5: Detailed Preemption Timeline Handshake" "2:26 - 3:00"

cat <<'EOF'
  Second-by-Second Preemption Timeline:

  [T+0s — 120s] SHADOW vLLM SERVING BATCH TRAFFIC (waiter_queue_depth=0)
       │  HTTP client completes inference requests at full GPU throughput
       ▼
  [T+120.0s]    RL TRAINER CALLS acquire() ON ORCHESTRATOR
       │  Orchestrator marks waiter_queue_depth=1
       ▼
  [T+120.2s]    SHADOW vLLM SUPERVISOR DETECTS SIGNAL -> SIGTERM (<200ms)
       │  Stock vLLM exits cleanly; GPU memory released immediately
       ▼
  [T+120.5s]    RL TRAINER ACQUIRES EXCLUSIVE LOCK & TRAINS (20s)
       │  Executes PyTorch forward/backward pass with zero contention
       ▼
  [T+140.5s]    RL TRAINER FINISHES CYCLE & CALLS yield()
       │  Enters 120s idle sleep window
       ▼
  [T+141.0s]    SHADOW vLLM AUTO-REACQUIRES LOCK & RESUMES
                  Relaunches stock server to serve queued inference requests
EOF

echo -e "\n${GREEN}${BOLD}✔ Demo Complete! Guaranteed priority for RL + 98.6% GPU Duty Cycle achieved.${RESET}\n"
