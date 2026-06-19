#!/usr/bin/env bash
# HARU Social AI 실행 스크립트 (System 2: Gemma 4 12B)
#
# 사용법:
#   ./launch_vla.sh           # 전체 파이프라인 (HITL 없음)
#   ./launch_vla.sh audio     # 전체 파이프라인 + 오디오 입력 (Phase 4.5)
#   ./launch_vla.sh hitl      # 전체 파이프라인 + HITL 로거 (데이터 수집)
#   ./launch_vla.sh brain     # brain_node 만 (Gemma 4 12B)
#   ./launch_vla.sh action    # action_node 만
#   ./launch_vla.sh vision    # vision_node 만
#   ./launch_vla.sh audio_only # audio_node 만

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

  hitl)
    echo "[LAUNCH] HITL 데이터 수집 모드: vision + action(hitl_mode) + brain + hitl_logger"

    ros2 run haru_vision vision_node &
    PID_VISION=$!

    ros2 run haru_action action_node --ros-args -p hitl_mode:=true &
    PID_ACTION=$!

    source "${VENV}/bin/activate"
    ros2 run haru_brain brain_node &
    PID_BRAIN=$!

    trap "kill ${PID_VISION} ${PID_ACTION} ${PID_BRAIN} 2>/dev/null; echo '[STOP] 종료'" SIGINT SIGTERM

    # hitl_node: 포그라운드 실행 (터미널 입력 필요)
    ros2 run haru_logger hitl_node
    wait
    ;;

  audio_only)
    echo "[LAUNCH] haru_audio_node (C270 마이크 캡처, VAD)"
    source "${VENV}/bin/activate"
    ros2 run haru_audio audio_node
    ;;

  audio)
    echo "[LAUNCH] Phase 4.5: vision + action + brain + audio (오디오 입력 포함)"

    ros2 run haru_vision vision_node &
    PID_VISION=$!

    ros2 run haru_action action_node &
    PID_ACTION=$!

    source "${VENV}/bin/activate"
    ros2 run haru_brain brain_node &
    PID_BRAIN=$!

    ros2 run haru_audio audio_node &
    PID_AUDIO=$!

    echo "[INFO] PIDs — vision:${PID_VISION}  action:${PID_ACTION}  brain:${PID_BRAIN}  audio:${PID_AUDIO}"
    echo "[INFO] Ctrl+C 로 전체 종료"

    trap "kill ${PID_VISION} ${PID_ACTION} ${PID_BRAIN} ${PID_AUDIO} 2>/dev/null; echo '[STOP] 전체 노드 종료'" SIGINT SIGTERM
    wait
    ;;

  all)
    echo "[LAUNCH] 전체 파이프라인: vision + action + brain (Gemma 4 12B)"

    ros2 run haru_vision vision_node &
    PID_VISION=$!

    ros2 run haru_action action_node &
    PID_ACTION=$!

    source "${VENV}/bin/activate"
    ros2 run haru_brain brain_node &
    PID_BRAIN=$!

    echo "[INFO] PIDs — vision:${PID_VISION}  action:${PID_ACTION}  brain:${PID_BRAIN}"
    echo "[INFO] Ctrl+C 로 전체 종료"

    trap "kill ${PID_VISION} ${PID_ACTION} ${PID_BRAIN} 2>/dev/null; echo '[STOP] 전체 노드 종료'" SIGINT SIGTERM
    wait
    ;;

  *)
    echo "사용법: $0 [brain|action|vision|all|audio|audio_only|hitl]"
    exit 1
    ;;
esac
