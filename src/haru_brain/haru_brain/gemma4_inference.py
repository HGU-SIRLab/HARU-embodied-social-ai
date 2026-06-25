"""
Gemma 4 12B Unified 추론 엔진 (HARU System 2)
- bf16 로드 (bitsandbytes 4-bit 비호환 → 폴백 구조 유지)
- 이미지 + 오디오(선택) + 텍스트 멀티모달 입력
- JSON 추출
"""

from __future__ import annotations

import copy
import json
import re
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Gemma4UnifiedForConditionalGeneration,
    BitsAndBytesConfig,
)

from .tom_prompt import TomPromptBuilder
from .adapter_manager import latest_adapter

logger = logging.getLogger(__name__)

MODEL_ID = 'google/gemma-4-12B-it'

# 관절 범위 (lo, hi, default)
JOINT_LIMITS = {
    'head_tilt':       (1500, 3086, 2048),
    'head_pan':        (1043, 3071, 2057),
    'head_roll':       (1630, 2452, 2041),
    'r_arm_pitch':     (1024, 2451, 1738),
    'l_arm_pitch':     (37,   1542,  790),
    'r_shoulder_roll': (1000, 2050, 1525),
    'r_elbow_pitch':   (2047, 3062, 2555),
    'l_shoulder_roll': (1047, 2056, 1552),
    'l_elbow_pitch':   (1021, 2007, 1514),
    'right_wheel':     (-300,  300,  0.0),
    'left_wheel':      (-300,  300,  0.0),
}

_NEUTRAL_ACTION = {k: float(v[2]) for k, v in JOINT_LIMITS.items()}


@dataclass
class HaruResponse:
    speech: str = ''
    emotion: str = 'neutral'
    expression_id: int = 0
    action: dict = field(default_factory=lambda: dict(_NEUTRAL_ACTION))
    duration: float = 2.5

    def to_command_dict(self) -> dict:
        return {
            'speech':        self.speech,
            'emotion':       self.emotion,
            'expression_id': self.expression_id,
            'action':        copy.copy(self.action),
            'duration':      self.duration,
        }


def _make_fallback() -> HaruResponse:
    """매번 새 인스턴스를 반환해 mutable 공유를 방지."""
    return HaruResponse(
        speech='잠시만요, 생각 중이에요.',
        emotion='thinking',
        expression_id=6,
        action=dict(_NEUTRAL_ACTION),
        duration=2.0,
    )


class Gemma4Inference:
    def __init__(self, model_id: str = MODEL_ID, load_in_4bit: bool = True):
        self._model_id = model_id
        self._load_in_4bit = load_in_4bit
        self._model = None
        self._processor = None
        self._tom = TomPromptBuilder()

    def load(self):
        logger.info(f'[Brain] Gemma 4 12B 로드 시작: {self._model_id}')

        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )

        if self._load_in_4bit:
            self._model = self._load_4bit()
        else:
            self._model = self._load_bf16()

        # PEFT LoRA 어댑터 적용 (학습된 어댑터가 있으면 자동 로드)
        self._try_load_adapter()

        self._model.eval()
        mem_gb = torch.cuda.memory_allocated() // 1024**3
        logger.info(f'[Brain] 모델 로드 완료. GPU 메모리 사용: {mem_gb}GB')

    def _try_load_adapter(self):
        adapter_path = latest_adapter()
        if adapter_path is None:
            return
        try:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(
                self._model,
                str(adapter_path),
                is_trainable=False,
            )
            logger.info(f'[Brain] LoRA 어댑터 로드 완료: {adapter_path.name}')
        except Exception as e:
            logger.warning(f'[Brain] LoRA 어댑터 로드 실패 (베이스 모델 유지): {e}')

    def _load_4bit(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        try:
            model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
                self._model_id,
                quantization_config=quant_config,
                dtype=torch.bfloat16,
                device_map='auto',
                trust_remote_code=True,
            )
            logger.info('[Brain] 4-bit 로드 성공.')
            return model
        except Exception as e:
            logger.warning(f'[Brain] 4-bit 로드 실패 ({e}), bf16으로 폴백')
            return self._load_bf16()

    def _load_bf16(self):
        logger.info('[Brain] bf16으로 로드 중... (~24GB)')
        return Gemma4UnifiedForConditionalGeneration.from_pretrained(
            self._model_id,
            dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )

    def infer(
        self,
        image: Image.Image,
        user_context: str = '',
        audio: np.ndarray | None = None,
        **kwargs,
    ) -> HaruResponse:
        if self._model is None:
            raise RuntimeError('load()를 먼저 호출하세요.')

        messages = self._tom.build_messages(image, user_context, audio=audio)

        try:
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors='pt',
            ).to(self._model.device)

            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=250,   # JSON ~200 토큰이면 충분
                    do_sample=False,      # greedy: sampling 오버헤드 제거
                    pad_token_id=self._processor.tokenizer.eos_token_id,
                )

            # 입력 토큰 제거 후 디코딩
            new_tokens = output_ids[:, inputs['input_ids'].shape[1]:]
            raw_text = self._processor.tokenizer.decode(
                new_tokens[0], skip_special_tokens=True
            )
            logger.debug(f'[Brain] 원본 출력: {raw_text[:200]!r}')

            response = self._parse_response(raw_text)
            # 성공한 경우에만 이력에 추가
            self._tom.add_assistant_turn(raw_text)
            return response

        except Exception as e:
            logger.error(f'[Brain] 추론 오류: {e}')
            return _make_fallback()

    def _parse_response(self, raw: str) -> HaruResponse:
        text = raw
        data = None

        # 1. ```json ... ``` 블록 우선 탐색
        m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 2. 중괄호 블록을 뒤에서부터 탐색 (thought 채널이 앞에 올 수 있으므로 역순)
        #    'action' 키를 포함한 첫 번째 유효 JSON을 사용
        if data is None:
            candidates = list(re.finditer(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', text, re.DOTALL))
            for m in reversed(candidates):
                try:
                    candidate = json.loads(m.group(1))
                    if 'action' in candidate:
                        data = candidate
                        break
                except json.JSONDecodeError:
                    continue

        if data is None:
            logger.warning(f'[Brain] JSON 없음, 원본: {text[:120]!r}')
            return _make_fallback()

        if not isinstance(data, dict):
            logger.warning(f'[Brain] JSON이 dict가 아님: {type(data)}')
            return _make_fallback()

        action = data.get('action', {})
        clamped_action = {}
        for joint, (lo, hi, default) in JOINT_LIMITS.items():
            raw_val = action.get(joint, default)
            try:
                clamped_action[joint] = float(max(lo, min(hi, float(raw_val))))
            except (ValueError, TypeError):
                clamped_action[joint] = float(default)

        expr_id = int(max(0, min(7, int(data.get('expression_id', 0)))))
        duration = float(max(0.5, min(10.0, float(data.get('duration', 2.5)))))
        speech = str(data.get('speech', ''))
        emotion = str(data.get('emotion', 'neutral'))

        return HaruResponse(
            speech=speech,
            emotion=emotion,
            expression_id=expr_id,
            action=clamped_action,
            duration=duration,
        )

    def reset_context(self):
        """에피소드 경계에서 SWM 이력 초기화."""
        self._tom.reset()
