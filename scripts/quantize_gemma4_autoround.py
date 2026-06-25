#!/usr/bin/env python3
"""
Gemma 4 12B W4A16 양자화 (auto_round 0.13.1) — 텍스트 전용 캘리브레이션

Gemma 4 Unified 특이 구조:
  - sliding_attention 레이어 (0-4, 6-10, ...): head_dim=256
  - full_attention 레이어 (5, 11, 17, ...): global_head_dim=512, partial_rotary_factor=0.25
  auto_round의 블록 단위 캘리브레이션이 position_embeddings를 레이어간 재사용하면서
  512 vs 256 차원 불일치가 발생 → apply_rotary_pos_emb monkey-patch로 해결

실행:
  source /home/herobot/robot_brain_workspace/haru_vla_env/bin/activate
  python3 scripts/quantize_gemma4_autoround.py 2>&1 | tee /tmp/quantize.log

완료 후:
  data/gemma4_autoround_w4a16/ 에 양자화 모델 저장
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
NSAMPLES   = 128
ITERS      = 0    # RTN 모드: ITERS=200(SignRound)은 scale 캡처 실패 문제 있음
BITS       = 4
GROUP_SIZE = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info('== Gemma 4 12B W4A16 양자화 시작 ==')
logger.info(f'모델 경로: {HF_MODEL}')
logger.info(f'출력 경로: {OUTPUT_DIR}')
logger.info(f'설정: bits={BITS}, group_size={GROUP_SIZE}, nsamples={NSAMPLES}, iters={ITERS}')

import torch
logger.info(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    logger.error('CUDA를 사용할 수 없습니다.')
    sys.exit(1)

# ── 1. auto_round MLLM 경로 우회 ──────────────────────────────────────────
# VLM calibration이 gemma3 템플릿과 충돌하므로 텍스트 전용 경로 강제
from auto_round.utils import model as _ar_model_utils
_ar_model_utils._LLM_ONLY_MODEL_TYPES.add('gemma4_unified')
_ar_model_utils._LLM_ONLY_MODEL_TYPES.add('gemma4')
logger.info('auto_round MLLM 경로 비활성화')

# ── 1b. Jetson PyTorch 2.5 set_submodule 버그 패치 ───────────────────────
# PyTorch 2.5 Jetson: set_submodule이 type(mod) is not nn.Module (exact check)
# → 모든 nn.Module 서브클래스에서 AttributeError → set_module 조용히 무시
# → QuantLinear 모델에 설치 안 됨 → release_layer_safely가 원본 weight 파괴
def _patched_set_module(model, key, new_module):
    atoms = key.split(".")
    name = atoms[-1]
    parent = model
    for atom in atoms[:-1]:
        parent = getattr(parent, atom)
    setattr(parent, name, new_module)

import auto_round.utils.model as _ar_model_mod
_ar_model_mod.set_module = _patched_set_module
try:
    import auto_round.utils.weight_handler as _ar_wh_mod
    _ar_wh_mod.set_module = _patched_set_module
except (ImportError, AttributeError):
    pass
logger.info('set_module Jetson PyTorch 2.5 버그 패치 적용')

# ── 2. Gemma 4 Unified RoPE 이종 차원 패치 ──────────────────────────────
# full_attention (head_dim=512) 레이어에서 sliding_attention (256) 캐시된
# position_embeddings를 재사용할 때 차원 불일치 → 안전한 RoPE 적용으로 해결.
# 캘리브레이션 전용 패치; 추론 시 원본 코드 복원 없이도 무관.
import transformers.models.gemma4_unified.modeling_gemma4_unified as _g4u_mod

_orig_rope = _g4u_mod.apply_rotary_pos_emb

def _safe_apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    """head_dim 불일치 시 RoPE 적용 가능한 차원만 회전, 나머지 pass-through.
    sliding_attention(256) ↔ full_attention(512) 이종 구조 대응."""
    if unsqueeze_dim is not None:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    rot_dim = min(x.shape[-1], cos.shape[-1])
    if rot_dim < x.shape[-1]:
        x_rot  = x[..., :rot_dim]
        x_pass = x[..., rot_dim:]
        x_rot  = (x_rot * cos[..., :rot_dim]) + (_g4u_mod.rotate_half(x_rot) * sin[..., :rot_dim])
        return torch.cat([x_rot, x_pass], dim=-1)
    return (x * cos) + (_g4u_mod.rotate_half(x) * sin)

_g4u_mod.apply_rotary_pos_emb = _safe_apply_rotary_pos_emb
logger.info('Gemma 4 Unified RoPE 이종 차원 패치 적용 (full_attention ↔ sliding_attention)')

# ── 2b. shared_kv_states=None 패치 ─────────────────────────────────────────
# 레이어 46(마지막 sliding) 및 47(마지막 full_attention)은 store_full_length_kv=True:
# auto_round 블록 단위 캘리브레이션 시 shared_kv_states가 None으로 전달되어
# `shared_kv_states[layer_type] = ...` 에서 TypeError 발생 → 빈 dict으로 초기화.
import functools as _functools

_OrigAttn = _g4u_mod.Gemma4UnifiedTextAttention
_orig_attn_fwd = _OrigAttn.forward

@_functools.wraps(_orig_attn_fwd)
def _safe_attn_forward(self, hidden_states, position_embeddings, attention_mask,
                       shared_kv_states, past_key_values=None, **kwargs):
    if shared_kv_states is None:
        shared_kv_states = {}
    return _orig_attn_fwd(self, hidden_states, position_embeddings, attention_mask,
                          shared_kv_states, past_key_values, **kwargs)

_OrigAttn.forward = _safe_attn_forward
logger.info('Gemma 4 Unified shared_kv_states=None 패치 적용 (레이어 46, 47)')

# ── 2c. WrapperLinear.unwrapper scale 캡처 패치 ─────────────────────────
# unwrapper() 직후 scale을 id→scale dict에 저장.
# save_quantized 전에 모델 레이어에 scale을 재주입해서 packing이 작동하게 함.
from auto_round.wrapper import WrapperLinear as _WrapperLinear
_orig_unwrapper = _WrapperLinear.unwrapper
_scale_by_id = {}  # {id(orig_layer): {'scale': ..., 'zp': ...}}

def _capturing_unwrapper(self, best_params):
    result = _orig_unwrapper(self, best_params)
    if hasattr(result, 'scale') and result.scale is not None:
        _scale_by_id[id(result)] = {
            'scale': result.scale.detach().clone() if isinstance(result.scale, torch.Tensor) else result.scale,
            'zp': (result.zp.detach().clone() if isinstance(getattr(result, 'zp', None), torch.Tensor)
                   else getattr(result, 'zp', None)),
        }
    return result

_WrapperLinear.unwrapper = _capturing_unwrapper

# ── 2d. pack_layer scale 없는 레이어 스킵 패치 ──────────────────────────
from auto_round.export.export_to_autoround import export as _ar_export
_ar_export.set_module = _patched_set_module  # export 모듈에도 패치 적용
_orig_pack_layer = _ar_export.pack_layer

_pack_ok = [0]
_pack_skip = [0]

def _safe_pack_layer(layer_name, model, backend, device):
    try:
        result = _orig_pack_layer(layer_name, model, backend, device)
        _pack_ok[0] += 1
        return result
    except AttributeError as e:
        if 'scale' in str(e):
            _pack_skip[0] += 1
            return
        raise

_ar_export.pack_layer = _safe_pack_layer
logger.info('auto_round unwrapper scale 캡처 + pack_layer 스킵 패치 적용')

# ── 3. 모델 로드 ─────────────────────────────────────────────────────────
logger.info('모델 로드 중...')
t0 = time.time()

from transformers import AutoProcessor

try:
    from transformers import Gemma4UnifiedForConditionalGeneration as ModelClass
    logger.info('Gemma4UnifiedForConditionalGeneration 사용')
except ImportError:
    from transformers import Gemma4ForConditionalGeneration as ModelClass
    logger.info('Gemma4ForConditionalGeneration 사용 (fallback)')

processor = AutoProcessor.from_pretrained(HF_MODEL, local_files_only=True)
tokenizer = processor.tokenizer

model = ModelClass.from_pretrained(
    HF_MODEL,
    dtype=torch.bfloat16,
    device_map='auto',
    local_files_only=True,
    trust_remote_code=True,
)
logger.info(f'모델 로드 완료: {time.time()-t0:.1f}s')
logger.info(f'GPU 메모리: {torch.cuda.memory_allocated()/1024**3:.1f}GB')

# ── 4. auto_round W4A16 양자화 ──────────────────────────────────────────
logger.info('auto_round W4A16 양자화 시작...')
from auto_round import AutoRound

quantizer = AutoRound(
    model,
    tokenizer,
    nsamples=NSAMPLES,
    iters=ITERS,
    bits=BITS,
    group_size=GROUP_SIZE,
    dtype='bfloat16',
    low_cpu_mem_usage=False,  # OffloadManager 비활성: convert_module_to_hp 후 reload로 qweight가 손실되는 버그 우회
    # processor/template 미전달 → 텍스트 전용 LLM 경로
)

t1 = time.time()
quantizer.quantize()
logger.info(f'양자화 완료: {time.time()-t1:.1f}s')

# scale 재주입: unwrapper 캡처된 scale을 레이어에 적용
from auto_round.utils.model import get_module as _get_mod
logger.info(f'[DEBUG] 캡처된 scale 수: {len(_scale_by_id)}')
_injected = 0
_not_found = 0
for _n in quantizer.layer_config:
    _layer = _get_mod(model, _n)
    _actual = getattr(_layer, 'orig_layer', _layer)
    if not hasattr(_actual, 'scale') or _actual.scale is None:
        _obj_id = id(_actual)
        if _obj_id in _scale_by_id:
            _info = _scale_by_id[_obj_id]
            _actual.scale = _info['scale']
            _actual.zp = _info.get('zp')
            _injected += 1
        else:
            _not_found += 1
logger.info(f'[DEBUG] scale 재주입: 성공={_injected} 미발견={_not_found}')

# ── 5. 저장 ─────────────────────────────────────────────────────────────
logger.info(f'저장 중: {OUTPUT_DIR}')
quantizer.save_quantized(OUTPUT_DIR, format='auto_round', inplace=True)
logger.info(f'[DEBUG] packing 결과: 성공={_pack_ok[0]} 스킵={_pack_skip[0]}')
processor.save_pretrained(OUTPUT_DIR)
logger.info('프로세서 저장 완료')

elapsed = time.time() - t0
logger.info(f'=== 완료! 총 소요 시간: {elapsed/60:.1f}분 ===')
logger.info(f'저장 위치: {OUTPUT_DIR}')
logger.info('다음 단계: brain_node.py가 자동으로 이 경로를 감지합니다.')
