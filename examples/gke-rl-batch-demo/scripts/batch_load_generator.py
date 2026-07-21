#!/usr/bin/env python3
"""
Continuous Batch Inference Client Load Generator
Queries Shadow vLLM API continuously and reports request throughput every 5 seconds.
"""
import time
import requests
import datetime

VLLM_ENDPOINT = "http://shadow-vllm-service.rl-batch-demo.svc.cluster.local:8000/v1/completions"
PROMPT = "Explain the advantages of cooperative accelerator time-slicing."

print("Continuous Batch Inference Load Generator started...", flush=True)

completed = 0
errors = 0
last_report = time.time()

while True:
    try:
        resp = requests.post(
            VLLM_ENDPOINT,
            json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "prompt": PROMPT, "max_tokens": 32},
            timeout=5,
        )
        if resp.status_code == 200:
            completed += 1
            choice_text = resp.json().get("choices", [{}])[0].get("text", "").strip().replace("\n", " ")
            ts_now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{ts_now}] COMPLETED INFERENCE QUERY #{completed}: \"{choice_text[:75]}...\"", flush=True)
        else:
            errors += 1
    except Exception:
        errors += 1
        time.sleep(0.5)
    
    if time.time() - last_report >= 5.0:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Throughput (last 5s): completed={completed}, errors/waiting={errors}", flush=True)
        completed = 0
        errors = 0
        last_report = time.time()
