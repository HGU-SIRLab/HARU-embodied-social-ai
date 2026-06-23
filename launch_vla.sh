#!/usr/bin/env bash
# HARU Social AI 실행 스크립트 (Triple-System: System 3 attention + System 2 brain + System 1 action)
#
# 사용법:
#   ./launch_vla.sh              # 전체 파이프라인 (Triple-System)
#   ./launch_vla.sh hitl         # 전체 파이프라인 + HITL 로거 (데이터 수집)
#   ./launch_vla.sh attention_only  # attention_node 단독 테스트
#   ./launch_vla.sh brain        # brain_node 만 (Gemma 4 12B)
#   ./launch_vla.sh action       # action_node 만
#   ./launch_vla.sh vision       # vision_node 만
#   ./launch_vla.sh audio_only   # audio_node 만

set -e

WORKSPACE="/home/herobot/robot_brain_workspace"
VENV="${WORKSPACE}/haru_vla_env"
VENV_SITE="${VENV}/lib/python3.10/site-packages"

# ── ROS2 환경 로드 ─────────────────────────────────────────────────────────────
source /opt/ros/humble/setup.bash
source "${WORKSPACE}/install/setup.bash"

# ── venv numpy가 시스템 numpy보다 먼저 로드되도록 PYTHONPATH 설정 ─────────────
export PYTHONPATH="${VENV_SITE}:${PYTHONPATH}"

MODE="${1:-all}"

case "$MODE" in
  brain)
    echo "[LAUNCH] haru_brain_node (Gemma 4 12B System 2)"
    source "${VENV}/bin/activate"
    ros2 run haru_brain brain_node
    ;;

  action)
    echo "[LAUNCH] haru_action_node (System 1, 50Hz)"
    ros2 run haru_action action_node
    ;;

  vision)
    echo "[LAUNCH] haru_vision_node"
    ros2 run haru_vision vision_node
    ;;

  audio_only)
    echo "[LAUNCH] haru_audio_node (C270 마이크 캡처, VAD)"
    source "${VENV}/bin/activate"
    ros2 run haru_audio audio_node
    ;;

  attention_only)
    echo "[LAUNCH] haru_attention_node 단독 테스트"
    echo "[INFO] 모니터링: ros2 topic echo /haru_attention/event"
    ros2 run haru_vision vision_node &
    PID_VISION=$!

    ros2 run haru_audio audio_node &
    PID_AUDIO=$!

    trap "kill ${PID_VISION} ${PID_AUDIO} 2>/dev/null; echo '[STOP] 종료'" SIGINT SIGTERM

    ros2 run haru_attention attention_node
    wait
    ;;

  hitl)
    echo "[LAUNCH] HITL 데이터 수집 모드 (Triple-System + hitl_logger)"

    ros2 run haru_vision vision_node &
    PID_VISION=$!

    ros2 run haru_attention attention_node &
    PID_ATTENTION=$!

    ros2 run haru_audio audio_node &
    PID_AUDIO=$!

    ros2 run haru_action action_node --ros-args -p hitl_mode:=true &
    PID_ACTION=$!

    source "${VENV}/bin/activate"
    ros2 run haru_brain brain_node &
    PID_BRAIN=$!

    trap "kill ${PID_VISION} ${PID_ATTENTION} ${PID_AUDIO} ${PID_ACTION} ${PID_BRAIN} 2>/dev/null; echo '[STOP] 종료'" SIGINT SIGTERM

    # hitl_node: 포그라운드 실행 (터미널 입력 필요)
    ros2 run haru_logger hitl_node
    wait
    ;;

  all)
    echo "[LAUNCH] Triple-System 전체: vision + attention + audio + brain + action"

    ros2 run haru_vision vision_node &
    PID_VISION=$!

    ros2 run haru_attention attention_node &
    PID_ATTENTION=$!

    ros2 run haru_audio audio_node &
    PID_AUDIO=$!

    ros2 run haru_action action_node &
    PID_ACTION=$!

    source "${VENV}/bin/activate"
    ros2 run haru_brain brain_node &
    PID_BRAIN=$!

    echo "[INFO] PIDs — vision:${PID_VISION}  attention:${PID_ATTENTION}  audio:${PID_AUDIO}  action:${PID_ACTION}  brain:${PID_BRAIN}"
    echo "[INFO] 상황 모니터링: ros2 topic echo /haru_attention/event"
    echo "[INFO] Ctrl+C 로 전체 종료"

    trap "kill ${PID_VISION} ${PID_ATTENTION} ${PID_AUDIO} ${PID_ACTION} ${PID_BRAIN} 2>/dev/null; echo '[STOP] 전체 노드 종료'" SIGINT SIGTERM
    wait
    ;;

  *)
    echo "사용법: $0 [all|hitl|attention_only|brain|action|vision|audio_only]"
    exit 1
    ;;
esac
