#!/usr/bin/env python3
"""
Simulated RL Sampler Script
Runs continuous rollout sampling (100% GPU duty cycle) on dedicated sampler GPU node.
"""
import time
import torch

print("Starting RL Sampler loop (100% GPU duty cycle)...", flush=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)

step = 0
while True:
    step += 1
    _ = torch.matmul(a, b)
    if step % 500 == 0:
        print(f"[Sampler] Completed {step} continuous sampling steps on {device}.", flush=True)
