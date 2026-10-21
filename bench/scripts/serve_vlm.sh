#!/usr/bin/env bash
# Serves the VLM judge (Gemma 4 31B, bf16) behind an OpenAI-compatible endpoint with vLLM, in the
# locked environment of bench/vlm_server/ (vLLM 0.25.1, torch 2.11.0+cu128; the paper's setup).
#
#   bash scripts/serve_vlm.sh                    # 1 GPU with >= 80GB (paper setup, TP=1)
#   DP=2 bash scripts/serve_vlm.sh               # one TP=1 replica per GPU, for throughput
#   TP=4 GPU_MEM_UTIL=0.84 bash scripts/serve_vlm.sh   # 4x 24GB GPUs (see README)
#
# The endpoint is http://localhost:${PORT}/v1 (pass it to the evaluator with --vlm-base-url).
set -euo pipefail

cd "$(dirname "$0")/../vlm_server"
uv sync --frozen --quiet

MODEL="${MODEL:-google/gemma-4-31B-it}"
REVISION="${REVISION:-842da3794eaa0b77d5f08bae87a17459d91ff475}"  # pinned checkpoint used for the paper
PORT="${PORT:-8000}"
TP="${TP:-1}"
DP="${DP:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# vLLM 0.25.1's compiled extension links against the CUDA 13 runtime shipped inside the venv.
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VLLM_USE_DEEP_GEMM=0

exec .venv/bin/vllm serve "$MODEL" \
    --revision "$REVISION" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image": 32, "video": 0}'
