#!/usr/bin/env python3
"""
Gemma 4 auto_round W4A16 양자화 모델 전체 기능 테스트
  - 모델 로드 & GPU 메모리 확인
  - 추론 속도 비교 (W4A16 vs bf16 기준치)
  - 출력 구조 검증 (JSON, 관절 범위, emotion 등)
  - 연속 대화 (SWM 컨텍스트 유지)
  - 엣지 케이스 (빈 이미지, 빈 컨텍스트)
  - brain_node 3단계 폴백 경로 자동 감지 확인

실행:
  source /home/herobot/robot_brain_workspace/haru_vla_env/bin/activate
  python3 scripts/test_autoround.py
"""

import sys, os, time, json
sys.path.insert(0, '/home/herobot/robot_brain_workspace/haru_vla_env/lib/python3.10/site-packages')
sys.path.insert(0, '/home/herobot/robot_brain_workspace/src/haru_brain')

import torch
import numpy as np
from PIL import Image

# ──────────────────────────────────────────────────────────────
PASS = '\033[92m✅ PASS\033[0m'
FAIL = '\033[91m❌ FAIL\033[0m'
WARN = '\033[93m⚠️  WARN\033[0m'
results = []

def check(name, cond, detail=''):
    tag = PASS if cond else FAIL
    results.append((name, cond))
    print(f'  {tag}  {name}' + (f' — {detail}' if detail else ''))
    return cond

def section(title):
    print(f'\n{"="*60}')
    print(f'  {title}')
    print('='*60)

# ──────────────────────────────────────────────────────────────
section('0. 사전 확인')

from haru_brain.gemma4_autoround_inference import quantized_model_exists, QUANTIZED_MODEL_DIR

check('양자화 모델 디렉토리 존재', os.path.isdir(QUANTIZED_MODEL_DIR), QUANTIZED_MODEL_DIR)
check('CUDA 사용 가능', torch.cuda.is_available(), f'device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음"}')

if not quantized_model_exists():
    print(f'\n  {FAIL}  양자화 모델 파일 없음 — 먼저 scripts/quantize_gemma4_autoround.py 실행 필요')
    sys.exit(1)

files = os.listdir(QUANTIZED_MODEL_DIR)
check('config.json 존재', 'config.json' in files)
has_weights = any(f.endswith(('.safetensors', '.bin')) for f in files)
check('.safetensors/.bin 가중치 파일 존재', has_weights, f'{sum(1 for f in files if f.endswith((".safetensors",".bin")))}개')

# ──────────────────────────────────────────────────────────────
section('1. 모델 로드')

from haru_brain.gemma4_autoround_inference import Gemma4AutoRoundInference

brain = Gemma4AutoRoundInference()
mem_before = torch.cuda.memory_allocated() / 1024**3

t_load = time.time()
try:
    brain.load()
    load_time = time.time() - t_load
    mem_after = torch.cuda.memory_allocated() / 1024**3
    mem_used = mem_after - mem_before
    check('로드 성공', True, f'{load_time:.1f}s')
    check('GPU 메모리 W4A16 (~6-8GB 예상)', 3 < mem_used < 15, f'{mem_used:.1f}GB 사용')
except Exception as e:
    check('로드 성공', False, str(e))
    print(f'\n모델 로드 실패 — 테스트 중단')
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
section('2. 추론 속도 & 출력 품질')

dummy_img = Image.fromarray(np.full((448, 896, 3), 128, dtype=np.uint8))

print('\n  [Test 2-1] 기본 추론 (사용자 가만히 앉아있는 상황)')
t0 = time.time()
try:
    resp = brain.infer(dummy_img, user_context='사용자가 카메라 앞에 조용히 앉아 있습니다.')
    elapsed = time.time() - t0
    check('추론 성공', True)
    check('추론 시간 < 30s', elapsed < 30, f'{elapsed:.1f}s')
    check('speech 필드 존재', isinstance(resp.speech, str), repr(resp.speech[:50]))
    check('emotion 유효 값', resp.emotion in {'neutral','joy','sadness','curiosity','surprise','empathy','excitement','concern','thinking'}, resp.emotion)
    check('expression_id 범위 0-7', 0 <= resp.expression_id <= 7, str(resp.expression_id))
    check('duration 범위 0.5-10', 0.5 <= resp.duration <= 10.0, f'{resp.duration}s')
    check('head_tilt 범위 1500-3086', 1500 <= resp.action.get('head_tilt', 0) <= 3086, str(resp.action.get('head_tilt')))
    check('right_wheel 범위 -300~300', -300 <= resp.action.get('right_wheel', 0) <= 300, str(resp.action.get('right_wheel')))
    print(f'\n  출력 미리보기:')
    print(f'    speech   : {resp.speech[:80]}')
    print(f'    emotion  : {resp.emotion}  expr_id={resp.expression_id}  dur={resp.duration}s')
    print(f'    head     : tilt={resp.action.get("head_tilt")} pan={resp.action.get("head_pan")}')
    print(f'    wheels   : R={resp.action.get("right_wheel")} L={resp.action.get("left_wheel")}')
except Exception as e:
    check('추론 성공', False, str(e))

# ──────────────────────────────────────────────────────────────
section('3. 연속 대화 (SWM 컨텍스트)')

print('\n  [Test 3-1] 첫 번째 대화')
resp1 = brain.infer(dummy_img, user_context='안녕, 오늘 기분이 어때?')
check('1번 응답 정상', resp1.speech != '' or resp1.emotion != 'neutral', f'speech={bool(resp1.speech)} emotion={resp1.emotion}')

print('\n  [Test 3-2] 두 번째 대화 (이전 맥락 이어지는지)')
resp2 = brain.infer(dummy_img, user_context='방금 한 말 기억해?')
check('2번 응답 정상', resp2 is not None, f'speech={resp2.speech[:40] if resp2.speech else "(침묵)"}')

# ──────────────────────────────────────────────────────────────
section('4. 엣지 케이스')

print('\n  [Test 4-1] 빈 컨텍스트')
brain.reset_context()
resp_empty = brain.infer(dummy_img, user_context='')
check('빈 컨텍스트 처리', resp_empty is not None, f'speech 길이={len(resp_empty.speech)}')

print('\n  [Test 4-2] 완전히 검정 이미지')
black_img = Image.fromarray(np.zeros((448, 896, 3), dtype=np.uint8))
resp_black = brain.infer(black_img, user_context='어두운 곳이에요.')
check('검정 이미지 처리', resp_black is not None, f'emotion={resp_black.emotion}')

print('\n  [Test 4-3] 슬픈 상황 — 공감 감정 표현 여부')
brain.reset_context()
resp_sad = brain.infer(dummy_img, user_context='사용자가 울고 있습니다. 눈물이 흐르고 있어요.')
empathy_emotions = {'sadness', 'empathy', 'concern'}
check('슬픔 상황 공감 감정', resp_sad.emotion in empathy_emotions or resp_sad.expression_id in {2, 5, 7}, f'emotion={resp_sad.emotion} expr_id={resp_sad.expression_id}')

# ──────────────────────────────────────────────────────────────
section('5. 속도 벤치마크 (5회 평균)')

brain.reset_context()
times = []
print('  5회 추론 중...')
for i in range(5):
    t = time.time()
    r = brain.infer(dummy_img, user_context=f'테스트 {i+1}번째 반복입니다.')
    times.append(time.time() - t)
    print(f'    [{i+1}/5] {times[-1]:.1f}s — speech={r.speech[:30]!r}')

avg = sum(times) / len(times)
fastest = min(times)
check(f'평균 추론 < 25s', avg < 25, f'평균={avg:.1f}s  최소={fastest:.1f}s')
print(f'\n  ┌─ 추론 속도 요약 ──────────────────')
print(f'  │  평균: {avg:.1f}s')
print(f'  │  최소: {fastest:.1f}s')
print(f'  │  최대: {max(times):.1f}s')
print(f'  │  bf16 기준치: ~20-30s (Jetson AGX Orin)')
print(f'  │  예상 W4A16: ~5-15s (3-4× 향상)')
print(f'  └───────────────────────────────────')

# ──────────────────────────────────────────────────────────────
section('6. brain_node 자동 감지 경로 확인')

try:
    from haru_brain.gemma4_autoround_inference import quantized_model_exists
    exists = quantized_model_exists()
    check('quantized_model_exists() = True', exists, QUANTIZED_MODEL_DIR)

    # brain_node 모듈 임포트 (ROS2 없이)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'brain_node',
        '/home/herobot/robot_brain_workspace/src/haru_brain/haru_brain/brain_node.py'
    )
    # brain_node는 rclpy 의존성이 있어서 직접 임포트하면 실패할 수 있음
    # 대신 모듈 파일 존재 + 코드 내용으로 확인
    with open('/home/herobot/robot_brain_workspace/src/haru_brain/haru_brain/brain_node.py') as f:
        bn_code = f.read()
    check('brain_node에 AutoRound 경로 존재', 'gemma4_autoround_inference' in bn_code, 'import 확인')
    check('brain_node 3단계 폴백 구조 존재', '_mode_tag' in bn_code and 'AutoRound-W4A16' in bn_code, '모드 태그 확인')
except Exception as e:
    check('brain_node 구조 확인', False, str(e))

# ──────────────────────────────────────────────────────────────
section('결과 요약')

total = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print(f'\n  총 {total}개 테스트: {PASS} {passed}개  {FAIL if failed else ""} {failed}개')
print()

if failed > 0:
    print('  실패 항목:')
    for name, ok in results:
        if not ok:
            print(f'    ❌ {name}')

if failed == 0:
    print('  모든 테스트 통과 — auto_round W4A16 양자화 모델 정상 동작!')
    print('  brain_node가 자동으로 이 모드를 선택합니다.')
else:
    print('  일부 테스트 실패 — 위 로그를 확인하세요.')

sys.exit(0 if failed == 0 else 1)
