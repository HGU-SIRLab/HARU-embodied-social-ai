#!/bin/bash
# vLLM serve 명령 - 컨테이너 내부에서 실행
exec /opt/venv/bin/vllm serve /model \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.75 \
    --limit-mm-per-prompt '{"image": 1}' \
    --trust-remote-code \
    --port 8000 \
    --max-num-seqs 4 \
    --served-model-name gemma4
