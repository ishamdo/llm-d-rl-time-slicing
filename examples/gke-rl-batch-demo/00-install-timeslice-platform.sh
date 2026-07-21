#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "==> Building Linux x86_64 binaries for orchestrator and snapshot-agent..."
mkdir -p "${REPO_ROOT}/bin"
GOOS=linux GOARCH=amd64 go build -o "${REPO_ROOT}/bin/llm-d-rl-time-slicing" "${REPO_ROOT}/cmd/acceleratororchestrator"
GOOS=linux GOARCH=amd64 go build -o "${REPO_ROOT}/bin/snapshot-agent" "${REPO_ROOT}/cmd/snapshot-agent"

echo "==> Ensuring timeslice-system namespace and binary server are running..."
kubectl create namespace timeslice-system --dry-run=client -o yaml | kubectl apply -f -
kubectl label namespace timeslice-system app.kubernetes.io/managed-by=Helm --overwrite
kubectl annotate namespace timeslice-system meta.helm.sh/release-name=timeslice meta.helm.sh/release-namespace=timeslice-system --overwrite
kubectl apply -f "${SCRIPT_DIR}/00a-binary-server.yaml"
kubectl wait --for=condition=Ready pod/timeslice-binary-server -n timeslice-system --timeout=60s

echo "==> Copying compiled binaries and Python SDK into timeslice-binary-server..."
kubectl cp "${REPO_ROOT}/bin/llm-d-rl-time-slicing" timeslice-system/timeslice-binary-server:/data/llm-d-rl-time-slicing
kubectl cp "${REPO_ROOT}/bin/snapshot-agent" timeslice-system/timeslice-binary-server:/data/snapshot-agent
kubectl cp "${REPO_ROOT}/timeslice-sdk.tar.gz" timeslice-system/timeslice-binary-server:/data/timeslice-sdk.tar.gz
kubectl exec -n timeslice-system timeslice-binary-server -- chmod +x /data/llm-d-rl-time-slicing /data/snapshot-agent

echo "==> Updating Helm dependencies for parent timeslice chart..."
helm dependency update "${REPO_ROOT}/deploy"

echo "==> Installing/Upgrading Timeslice Platform (Orchestrator + Snapshot Agent + DRA Driver)..."
helm upgrade --install timeslice "${REPO_ROOT}/deploy" \
  -f "${REPO_ROOT}/deploy/values-gke.yaml" \
  --namespace timeslice-system \
  --create-namespace

echo "==> Verifying rollout status in timeslice-system namespace..."
kubectl get pods -n timeslice-system
