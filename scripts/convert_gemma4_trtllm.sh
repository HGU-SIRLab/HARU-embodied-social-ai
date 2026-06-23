#!/usr/bin/env bash
# Gemma 4 12B → TRT-LLM W4A16 엔진 변환
# Docker 이미지 빌드 완료 후 실행
#
# 출력:
#   data/trtllm/checkpoint/   ← INT4 양자화 체크포인트
#   data/trtllm/engine/       ← TRT-LLM 실행 엔진
#   data/trtllm/visual_engine/ ← 비전 인코더 엔진
#
# 소요 시간: 1~3시간 (INT4 양자화 포함)

set -e

WORKSPACE="/home/herobot/robot_brain_workspace"
HF_MODEL="/home/herobot/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/5926caa4ec0cac5cbfadaf4077420520de1d5205"
OUTPUT_CHECKPOINT="$WORKSPACE/data/trtllm/checkpoint"
OUTPUT_ENGINE="$WORKSPACE/data/trtllm/engine"
OUTPUT_VISUAL="$WORKSPACE/data/trtllm/visual_engine"
LOG_FILE="/tmp/trtllm_convert.log"

# Docker 이미지 이름 자동 감지
DOCKER_IMAGE=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep tensorrt_llm | grep -v builder | head -1)
if [ -z "$DOCKER_IMAGE" ]; then
    echo "[ERROR] TRT-LLM Docker 이미지 없음. build_trtllm_docker.sh 먼저 실행하세요."
    exit 1
fi

echo "================================================================"
echo " Gemma 4 12B → TRT-LLM W4A16 변환"
echo " 이미지: $DOCKER_IMAGE"
echo " 모델:   $HF_MODEL"
echo " 출력:   $OUTPUT_ENGINE"
echo "================================================================"

mkdir -p "$OUTPUT_CHECKPOINT" "$OUTPUT_ENGINE" "$OUTPUT_VISUAL"

# ── Docker 내부 변환 스크립트 ────────────────────────────────────────
cat > /tmp/trtllm_convert_inner.sh << 'INNER_EOF'
#!/usr/bin/env bash
set -ex

HF_MODEL="$1"
OUTPUT_CHECKPOINT="$2"
OUTPUT_ENGINE="$3"
OUTPUT_VISUAL="$4"

TRT_LLM_DIR="/opt/TensorRT-LLM"

# ── TRT-LLM 예제 경로 자동 탐색 ─────────────────────────────────────
GEMMA_EXAMPLE=""
for path in \
    "$TRT_LLM_DIR/examples/models/core/gemma" \
    "$TRT_LLM_DIR/examples/gemma" \
    "$TRT_LLM_DIR/examples/models/gemma"; do
    if [ -d "$path" ]; then
        GEMMA_EXAMPLE="$path"
        break
    fi
done

MULTIMODAL_EXAMPLE=""
for path in \
    "$TRT_LLM_DIR/examples/multimodal" \
    "$TRT_LLM_DIR/examples/models/core/multimodal"; do
    if [ -d "$path" ]; then
        MULTIMODAL_EXAMPLE="$path"
        break
    fi
done

echo "=== Gemma example dir: $GEMMA_EXAMPLE ==="
echo "=== Multimodal example dir: $MULTIMODAL_EXAMPLE ==="

# ── Step 1: HF → TRT-LLM 체크포인트 변환 (INT4 W4A16 양자화) ────────
echo "[1/3] HF 체크포인트 → TRT-LLM 형식 변환 (INT4 weight-only)"

CONVERT_SCRIPT=""
for s in \
    "$GEMMA_EXAMPLE/convert_checkpoint.py" \
    "$TRT_LLM_DIR/examples/convert_checkpoint.py"; do
    if [ -f "$s" ]; then
        CONVERT_SCRIPT="$s"
        break
    fi
done

if [ -n "$CONVERT_SCRIPT" ]; then
    python3 "$CONVERT_SCRIPT" \
        --model_dir "$HF_MODEL" \
        --output_dir "$OUTPUT_CHECKPOINT" \
        --dtype bfloat16 \
        --use_weight_only \
        --weight_only_precision int4
else
    # TRT-LLM 1.x 고수준 API 사용
    echo "[INFO] convert_checkpoint.py 없음 — 고수준 API로 진행"
    python3 - << 'PYEOF'
from tensorrt_llm import LLM, BuildConfig, QuantConfig, QuantAlgo
import os

hf_model = os.environ.get('HF_MODEL')
output_ckpt = os.environ.get('OUTPUT_CHECKPOINT')

quant_config = QuantConfig(
    quant_algo=QuantAlgo.W4A16,  # INT4 weights, BF16 activations
)

build_config = BuildConfig(
    max_input_len=3840,
    max_seq_len=4096,
    max_batch_size=1,
)

llm = LLM(
    model=hf_model,
    quant_config=quant_config,
    build_config=build_config,
)
llm.save(output_ckpt)
PYEOF
fi

# ── Step 2: 엔진 빌드 ────────────────────────────────────────────────
echo "[2/3] TRT-LLM 엔진 빌드 (max_batch_size=1, max_seq_len=4096)"

if command -v trtllm-build &>/dev/null; then
    trtllm-build \
        --checkpoint_dir "$OUTPUT_CHECKPOINT" \
        --output_dir "$OUTPUT_ENGINE" \
        --gemm_plugin bfloat16 \
        --gpt_attention_plugin bfloat16 \
        --max_batch_size 1 \
        --max_input_len 3840 \
        --max_seq_len 4096 \
        --max_num_tokens 4096 \
        --use_paged_context_fmha enable \
        --multiple_profiles enable
else
    echo "[ERROR] trtllm-build 명령 없음. TRT-LLM 설치를 확인하세요."
    exit 1
fi

# ── Step 3: 비전 인코더 빌드 ─────────────────────────────────────────
echo "[3/3] 비전 인코더 TRT 엔진 빌드"

VISUAL_SCRIPT=""
for s in \
    "$MULTIMODAL_EXAMPLE/build_visual_engine.py" \
    "$TRT_LLM_DIR/examples/multimodal/build_visual_engine.py"; do
    if [ -f "$s" ]; then
        VISUAL_SCRIPT="$s"
        break
    fi
done

if [ -n "$VISUAL_SCRIPT" ]; then
    python3 "$VISUAL_SCRIPT" \
        --model_type gemma4 \
        --model_path "$HF_MODEL" \
        --output_dir "$OUTPUT_VISUAL" \
        --max_batch_size 1 \
        || python3 "$VISUAL_SCRIPT" \
            --model_type gemma \
            --model_path "$HF_MODEL" \
            --output_dir "$OUTPUT_VISUAL" \
            --max_batch_size 1
else
    echo "[WARN] 비전 인코더 빌드 스크립트 없음 — 언어 모델만 사용"
fi

echo ""
echo "=== 변환 완료! ==="
echo "  엔진:        $OUTPUT_ENGINE"
echo "  비전 엔진:   $OUTPUT_VISUAL"
INNER_EOF

chmod +x /tmp/trtllm_convert_inner.sh

# Docker 컨테이너 내에서 변환 실행
docker run --rm \
    --gpus all \
    --ipc=host \
    -v "/home/herobot:/home/herobot" \
    -v "/tmp:/tmp" \
    -e "HF_MODEL=$HF_MODEL" \
    -e "OUTPUT_CHECKPOINT=$OUTPUT_CHECKPOINT" \
    "$DOCKER_IMAGE" \
    bash /tmp/trtllm_convert_inner.sh \
        "$HF_MODEL" \
        "$OUTPUT_CHECKPOINT" \
        "$OUTPUT_ENGINE" \
        "$OUTPUT_VISUAL" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "[$(date)] 변환 완료!"
echo "다음 단계:"
echo "  bash /home/herobot/robot_brain_workspace/scripts/run_trtllm_server.sh"
