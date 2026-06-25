#!/usr/bin/env bash
# vLLM 추론 서버 실행 (OpenAI 호환 API, 포트 8000)
# Gemma 4 W4A16 AutoRound + vLLM + Marlin + CUDAGraph
# 속도: ~19 tok/s (Phase 5.8 목표 달성)
#
# 서버 확인:
#   curl http://localhost:8000/v1/models
#
# 중단:
#   docker stop haru_vllm_server

set -e

WORKSPACE="/home/herobot/robot_brain_workspace"
# 패치된 config.json(architectures=Gemma4ForConditionalGeneration)이 포함된 모델 디렉토리
MODEL_DIR="${WORKSPACE}/data/gemma4_vllm_patched"
# gemma4_mm.py 패치 파일 (Gemma4UnifiedVisionEmbedder + unified weight mapper)
MM_PATCH="${WORKSPACE}/scripts/gemma4_mm_patch.py"
# torch.compile 캐시 (재시작 시 컴파일 90s 절약)
COMPILE_CACHE="${WORKSPACE}/data/vllm_compile_cache"
PORT=8000
CONTAINER_NAME="haru_vllm_server"
GPU_MEMORY_UTIL=0.70
DOCKER_IMAGE="vllm:r36.5.tegra-aarch64-cu126-22.04-vllm"

# 모델 디렉토리 확인
if [ ! -d "$MODEL_DIR" ]; then
    echo "[ERROR] 패치된 모델 없음: $MODEL_DIR"
    echo "       scripts/run_vllm_server.sh 첫 실행 전 gemma4_vllm_patched 디렉토리 생성 필요"
    exit 1
fi

# gemma4_mm 패치 파일 확인
if [ ! -f "$MM_PATCH" ]; then
    echo "[ERROR] gemma4_mm 패치 없음: $MM_PATCH"
    exit 1
fi

mkdir -p "$COMPILE_CACHE"

# 기존 컨테이너 정리
sg docker -c "docker stop ${CONTAINER_NAME} 2>/dev/null || true"
sg docker -c "docker rm   ${CONTAINER_NAME} 2>/dev/null || true"

VLLM_MM_PATH="/opt/venv/lib/python3.10/site-packages/vllm/model_executor/models/gemma4_mm.py"
SERVE_CMD="${WORKSPACE}/scripts/vllm_serve_cmd.sh"

echo "[INFO] vLLM 서버 시작... (포트 ${PORT})"
echo "[INFO] 모델: ${MODEL_DIR}"
echo "[INFO] 첫 시작: torch.compile 약 90초, 이후: 캐시 사용"

sg docker -c "docker run -d \
    --name ${CONTAINER_NAME} \
    --runtime nvidia \
    --gpus all \
    --network host \
    --shm-size=8g \
    -v ${MODEL_DIR}:/model:ro \
    -v ${MM_PATCH}:${VLLM_MM_PATH}:ro \
    -v ${COMPILE_CACHE}:/root/.cache/vllm/torch_compile_cache \
    -v ${SERVE_CMD}:/opt/vllm_serve_cmd.sh:ro \
    ${DOCKER_IMAGE} \
    bash /opt/vllm_serve_cmd.sh"

echo "[INFO] 컨테이너 시작됨. 서버 준비까지 3~5분 대기 (최초: compile 포함)..."
sleep 10

# 준비될 때까지 폴링 (최대 360초 = 6분)
for i in $(seq 1 36); do
    if curl -sf http://localhost:${PORT}/v1/models > /dev/null 2>&1; then
        echo "[OK] vLLM 서버 준비 완료!"
        curl -s http://localhost:${PORT}/v1/models | python3 -m json.tool
        break
    fi
    echo "[WAIT] ${i}/36 (${i}x10s)..."
    sleep 10
done
