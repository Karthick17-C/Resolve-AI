#!/usr/bin/env bash
set -euo pipefail

vllm serve "${MODEL_PATH:-./resolve-final-model}" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 1536 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.80 \
  --enforce-eager \
  --disable-custom-all-reduce \
  --api-key "${RESOLVE_API_KEY:-sk-resolve-secret-key-98765}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes