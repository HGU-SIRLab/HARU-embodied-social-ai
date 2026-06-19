"""
Gemma 4 12B 추론 빠른 테스트 (ROS2 없이)
사용: python scripts/test_gemma4.py [--fp16]
  --fp16  : 4-bit 대신 fp16으로 로드 (메모리 ~24GB, 정확도 최대)
"""

import sys
import argparse
sys.path.insert(0, '/home/herobot/robot_brain_workspace/haru_vla_env/lib/python3.10/site-packages')
sys.path.insert(0, '/home/herobot/robot_brain_workspace/src/haru_brain')

from PIL import Image
import numpy as np
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='4-bit 대신 fp16 로드')
    args = parser.parse_args()

    load_in_4bit = not args.fp16
    print(f'[TEST] Gemma 4 12B 추론 테스트 ({"4-bit" if load_in_4bit else "fp16"})')

    from haru_brain.gemma4_inference import Gemma4Inference

    brain = Gemma4Inference(load_in_4bit=load_in_4bit)
    t_load = time.time()
    brain.load()
    print(f'[TEST] 모델 로드 완료 ({time.time() - t_load:.1f}s)\n')

    # 테스트 1: 더미 이미지 (회색)
    dummy = Image.fromarray(np.full((448, 896, 3), 128, dtype=np.uint8))
    print('[TEST 1] 더미 이미지 추론...')
    t0 = time.time()
    resp = brain.infer(dummy, user_context='사용자가 카메라 앞에 조용히 앉아 있습니다.')
    print(f'  추론 시간: {time.time() - t0:.1f}s')
    print(f'  speech     : {resp.speech}')
    print(f'  emotion    : {resp.emotion}')
    print(f'  expression : {resp.expression_id}')
    print(f'  duration   : {resp.duration}')
    print(f'  head_tilt  : {resp.action.get("head_tilt")}')
    print(f'  head_pan   : {resp.action.get("head_pan")}')
    print(f'  r_arm_pitch: {resp.action.get("r_arm_pitch")}')
    print(f'  wheels     : R={resp.action.get("right_wheel")}  L={resp.action.get("left_wheel")}')

    # 테스트 2: 연속 대화 (SWM 이력 유지 확인)
    print('\n[TEST 2] 연속 대화 테스트...')
    t0 = time.time()
    resp2 = brain.infer(dummy, user_context='사용자가 슬퍼 보입니다.')
    print(f'  추론 시간: {time.time() - t0:.1f}s')
    print(f'  speech: {resp2.speech}')
    print(f'  emotion: {resp2.emotion}')

    print('\n[TEST] 완료')


if __name__ == '__main__':
    main()
