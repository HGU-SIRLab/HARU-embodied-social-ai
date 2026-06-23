#!/usr/bin/env bash
# TensorRT-LLM 1.2.0 Docker 이미지 빌드 (Jetson AGX Orin, CUDA 12.6, SM87)
# 소요 시간: 4~8 시간 (최초 1회)
# 로그: /tmp/trtllm_build.log
#
# 완료 후 이미지 확인:
#   docker images | grep tensorrt_llm
#
# 그 다음:
#   bash scripts/convert_gemma4_trtllm.sh
#   bash scripts/run_trtllm_server.sh

set -e

JETSON_CONTAINERS_DIR="/home/herobot/herobot_ws/jetson-containers"
LOG_FILE="/tmp/trtllm_build.log"

echo "================================================================"
echo " TensorRT-LLM 1.2.0 Jetson 빌드 시작"
echo " 로그: $LOG_FILE"
echo " 완료까지 4~8시간 소요 (CPU 코어 수에 따라 다름)"
echo "================================================================"

if [ ! -d "$JETSON_CONTAINERS_DIR" ]; then
    echo "[ERROR] jetson-containers 디렉토리 없음: $JETSON_CONTAINERS_DIR"
    exit 1
fi

cd "$JETSON_CONTAINERS_DIR"

# jetson-containers가 L4T 36.5.0을 모를 경우를 대비해 환경변수 명시
export L4T_VERSION="36.5.0"
export JETPACK_VERSION="6.5"
export CUDA_VERSION="12.6"

echo "[$(date)] 빌드 시작..." | tee "$LOG_FILE"

python3 -m jetson_containers.build tensorrt_llm 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "[$(date)] 빌드 완료!"
echo ""
echo "다음 단계:"
echo "  bash /home/herobot/robot_brain_workspace/scripts/convert_gemma4_trtllm.sh"
