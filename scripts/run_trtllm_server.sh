#!/usr/bin/env bash
# TRT-LLM 추론 서버 실행 (OpenAI 호환 API, 포트 8000)
# brain_node.py가 이 서버에 HTTP로 연결해 추론을 요청한다.
#
# 전제: convert_gemma4_trtllm.sh 실행 완료
#
# 서버 확인:
#   curl http://localhost:8000/v1/models
#
# 중단:
#   docker stop haru_trtllm_server

set -e

WORKSPACE="/home/herobot/robot_brain_workspace"
ENGINE_DIR="$WORKSPACE/data/trtllm/engine"
VISUAL_DIR="$WORKSPACE/data/trtllm/visual_engine"
HF_MODEL="/home/herobot/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/5926caa4ec0cac5cbfadaf4077420520de1d5205"
PORT=8000
CONTAINER_NAME="haru_trtllm_server"

# Docker 이미지 자동 감지
DOCKER_IMAGE=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep tensorrt_llm | grep -v builder | head -1)
if [ -z "$DOCKER_IMAGE" ]; then
    echo "[ERROR] TRT-LLM Docker 이미지 없음."
    exit 1
fi

if [ ! -d "$ENGINE_DIR" ] || [ -z "$(ls -A $ENGINE_DIR 2>/dev/null)" ]; then
    echo "[ERROR] 엔진 없음: $ENGINE_DIR"
    echo "       convert_gemma4_trtllm.sh 먼저 실행하세요."
    exit 1
fi

# 기존 컨테이너 정리
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm   "$CONTAINER_NAME" 2>/dev/null || true

echo "================================================================"
echo " HARU TRT-LLM 서버 시작"
echo " 이미지: $DOCKER_IMAGE"
echo " 엔진:   $ENGINE_DIR"
echo " 포트:   $PORT"
echo "================================================================"

# 비전 인코더 옵션 (있으면 포함)
VISUAL_ARGS=""
if [ -d "$VISUAL_DIR" ] && [ -n "$(ls -A $VISUAL_DIR 2>/dev/null)" ]; then
    VISUAL_ARGS="--visual_engine_dir /trtllm/visual_engine"
fi

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    --network host \
    -v "$ENGINE_DIR:/trtllm/engine:ro" \
    -v "$VISUAL_DIR:/trtllm/visual_engine:ro" \
    -v "$HF_MODEL:/trtllm/hf_model:ro" \
    "$DOCKER_IMAGE" \
    bash -c "
        # 서버 실행 (trtllm-serve → fallback to python -m)
        if command -v trtllm-serve &>/dev/null; then
            trtllm-serve /trtllm/engine \
                --tokenizer /trtllm/hf_model \
                $VISUAL_ARGS \
                --host 0.0.0.0 --port $PORT \
                --max_beam_width 1
        else
            python3 -m tensorrt_llm.serve \
                --engine_dir /trtllm/engine \
                --tokenizer_dir /trtllm/hf_model \
                $VISUAL_ARGS \
                --host 0.0.0.0 --port $PORT
        fi
    "

echo ""
echo "[$(date)] 서버 시작됨 (컨테이너: $CONTAINER_NAME)"
echo ""
echo "서버 준비 완료 대기 중..."
for i in $(seq 1 30); do
    sleep 2
    if curl -sf "http://localhost:$PORT/v1/models" &>/dev/null; then
        echo ""
        echo "✅ 서버 준비 완료!"
        curl -s "http://localhost:$PORT/v1/models" | python3 -m json.tool 2>/dev/null || true
        break
    fi
    echo -n "."
    if [ $i -eq 30 ]; then
        echo ""
        echo "⚠️  60초 내 응답 없음. 로그 확인:"
        echo "   docker logs $CONTAINER_NAME"
    fi
done

echo ""
echo "관리 명령:"
echo "  docker logs -f $CONTAINER_NAME   # 실시간 로그"
echo "  docker stop $CONTAINER_NAME      # 서버 중단"
echo "  curl http://localhost:$PORT/v1/models"
