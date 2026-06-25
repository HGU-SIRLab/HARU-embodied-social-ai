#!/usr/bin/env bash
# vLLM 추론 서버 실행 (OpenAI 호환 API, 포트 8000)
# Gemma 4 W4A16 AutoRound(GPTQ packing) 모델 사용
#
# 서버 확인:
#   curl http://localhost:8000/v1/models
#
# 중단:
#   docker stop haru_vllm_server

set -e

WORKSPACE="/home/herobot/robot_brain_workspace"
MODEL_DIR="${WORKSPACE}/data/gemma4_autoround_w4a16"
PORT=8000
CONTAINER_NAME="haru_vllm_server"
GPU_MEMORY_UTIL=0.70

# Docker 이미지 자동 감지
DOCKER_IMAGE=$(sg docker -c "docker images --format '{{.Repository}}:{{.Tag}}'" | grep "vllm" | grep -v builder | grep -v "r36" | head -1)
if [ -z "$DOCKER_IMAGE" ]; then
    DOCKER_IMAGE=$(sg docker -c "docker images --format '{{.Repository}}:{{.Tag}}'" | grep "vllm" | grep -v builder | head -1)
fi
if [ -z "$DOCKER_IMAGE" ]; then
    echo "[ERROR] vLLM Docker 이미지 없음. 먼저 jetson-containers build vllm 실행"
    exit 1
fi
echo "[INFO] vLLM 이미지: ${DOCKER_IMAGE}"

# 모델 디렉토리 확인
if [ ! -d "$MODEL_DIR" ]; then
    echo "[ERROR] 모델 없음: $MODEL_DIR"
    exit 1
fi

# 기존 컨테이너 정리
sg docker -c "docker stop ${CONTAINER_NAME} 2>/dev/null || true"
sg docker -c "docker rm   ${CONTAINER_NAME} 2>/dev/null || true"

echo "[INFO] vLLM 서버 시작... (포트 ${PORT})"

sg docker -c "docker run -d \
    --name ${CONTAINER_NAME} \
    --runtime nvidia \
    --gpus all \
    --network host \
    --shm-size=8g \
    -v ${MODEL_DIR}:/model:ro \
    -v ${WORKSPACE}/scripts/gemma4_server.py:/opt/gemma4_server.py:ro \
    ${DOCKER_IMAGE} \
    python3 /opt/gemma4_server.py"

echo "[INFO] 컨테이너 시작됨. 서버 준비까지 30~60초 대기..."
sleep 5

# 준비될 때까지 폴링 (최대 120초)
for i in $(seq 1 24); do
    if curl -sf http://localhost:${PORT}/v1/models > /dev/null 2>&1; then
        echo "[OK] vLLM 서버 준비 완료!"
        curl -s http://localhost:${PORT}/v1/models | python3 -m json.tool
        break
    fi
    echo "[WAIT] ${i}/24 대기 중..."
    sleep 5
done
