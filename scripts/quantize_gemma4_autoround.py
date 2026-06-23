#!/usr/bin/env python3
"""
Gemma 4 12B W4A16 양자화 (auto_round 0.13.1)
- W4: 4-bit weight quantization
- A16: BF16 activation
- group_size=128 (정확도/속도 균형)
- 소요 시간: 약 1-3시간

실행:
  source /home/herobot/robot_brain_workspace/haru_vla_env/bin/activate
  python3 scripts/quantize_gemma4_autoround.py 2>&1 | tee /tmp/quantize.log

완료 후:
  data/gemma4_autoround_w4a16/ 에 양자화 모델 저장
  brain_node.py가 이 경로를 자동으로 감지해 사용
"""

import os, sys, time, logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

WORKSPACE  = '/home/herobot/robot_brain_workspace'
HF_MODEL   = '/home/herobot/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/5926caa4ec0cac5cbfadaf4077420520de1d5205'
OUTPUT_DIR = os.path.join(WORKSPACE, 'data', 'gemma4_autoround_w4a16')
NSAMPLES   = 128    # 캘리브레이션 샘플 수 (많을수록 정확 but 느림)
ITERS      = 200    # 최적화 반복 횟수
BITS       = 4      # 양자화 비트
GROUP_SIZE = 128    # 그룹 크기

os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info('== Gemma 4 12B W4A16 양자화 시작 ==')
logger.info(f'모델 경로: {HF_MODEL}')
logger.info(f'출력 경로: {OUTPUT_DIR}')
logger.info(f'설정: bits={BITS}, group_size={GROUP_SIZE}, nsamples={NSAMPLES}, iters={ITERS}')

import torch
logger.info(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    logger.error('CUDA를 사용할 수 없습니다. GPU가 필요합니다.')
    sys.exit(1)

logger.info('transformers 모델 로드 중...')
t0 = time.time()

from transformers import AutoProcessor

try:
    from transformers import Gemma4ForConditionalGeneration
    MODEL_CLASS = Gemma4ForConditionalGeneration
    logger.info('Gemma4ForConditionalGeneration 사용')
except ImportError:
    from transformers import Gemma4UnifiedForConditionalGeneration
    MODEL_CLASS = Gemma4UnifiedForConditionalGeneration
    logger.info('Gemma4UnifiedForConditionalGeneration 사용 (fallback)')

processor = AutoProcessor.from_pretrained(HF_MODEL, local_files_only=True)
tokenizer = processor.tokenizer

model = MODEL_CLASS.from_pretrained(
    HF_MODEL,
    dtype=torch.bfloat16,
    device_map='auto',
    local_files_only=True,
    trust_remote_code=True,
)
logger.info(f'모델 로드 완료: {time.time()-t0:.1f}s')
logger.info(f'GPU 메모리: {torch.cuda.memory_allocated()/1024**3:.1f}GB')

logger.info('auto_round 양자화 시작...')
from auto_round import AutoRound

quantizer = AutoRound(
    model,
    tokenizer,
    nsamples=NSAMPLES,
    iters=ITERS,
    bits=BITS,
    group_size=GROUP_SIZE,
    dtype='bfloat16',
    # VLM 설정: 이미지 임베딩 레이어는 양자화하지 않음
    # (텍스트 트랜스포머 블록만 양자화)
)

t1 = time.time()
quantizer.quantize()
logger.info(f'양자화 완료: {time.time()-t1:.1f}s')

logger.info(f'양자화 모델 저장 중: {OUTPUT_DIR}')
quantizer.save_quantized(
    OUTPUT_DIR,
    format='autoround',
    inplace=False,
)

# processor도 함께 저장
processor.save_pretrained(OUTPUT_DIR)
logger.info(f'프로세서 저장 완료')

elapsed = time.time() - t0
logger.info(f'=== 완료! 총 소요 시간: {elapsed/60:.1f}분 ===')
logger.info(f'저장 위치: {OUTPUT_DIR}')
logger.info('')
logger.info('다음 단계: brain_node.py가 자동으로 이 경로를 감지합니다.')
logger.info('수동 확인: python3 scripts/test_gemma4.py')
