"""
Gemma 4 12B auto_round W4A16 양자화 추론 엔진 (HARU System 2 — 중속 경로)

TRT-LLM 서버가 없을 때 bf16 대비 ~3-5× 빠른 양자화 추론.
인터페이스는 Gemma4Inference와 동일 → brain_node가 투명하게 교체 가능.

quantize_gemma4_autoround.py 실행 완료 후 사용 가능.
data/gemma4_autoround_w4a16/ 경로에 양자화 모델이 있어야 함.
"""

from __future__ import annotations

import os
import copy
import json
import re
import logging
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .tom_prompt import TomPromptBuilder
from .adapter_manager import latest_adapter
from .gemma4_inference import HaruResponse, JOINT_LIMITS, _NEUTRAL_ACTION, _make_fallback

logger = logging.getLogger(__name__)

QUANTIZED_MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..', 'data', 'gemma4_autoround_w4a16'
)
QUANTIZED_MODEL_DIR = os.path.normpath(QUANTIZED_MODEL_DIR)


def quantized_model_exists() -> bool:
    """양자화 모델 디렉토리가 존재하고 파일이 있으면 True."""
    if not os.path.isdir(QUANTIZED_MODEL_DIR):
        return False
    files = os.listdir(QUANTIZED_MODEL_DIR)
    return any(f.endswith(('.safetensors', '.bin', 'config.json')) for f in files)


class Gemma4AutoRoundInference:
    """
    auto_round W4A16 4-bit 양자화 모델 추론.
    Gemma4Inference와 동일한 public interface:
      load() / infer(image, user_context, audio) / reset_context()
    """

    def __init__(self, model_dir: str = QUANTIZED_MODEL_DIR):
        self._model_dir = model_dir
        self._model = None
        self._processor = None
        self._tom = TomPromptBuilder()

    def load(self):
        logger.info(f'[AutoRound] W4A16 양자화 모델 로드: {self._model_dir}')

        from auto_round.inference.convert_model import post_init
        from transformers import AutoProcessor

        try:
            from transformers import Gemma4UnifiedForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import Gemma4ForConditionalGeneration as ModelClass

        self._processor = AutoProcessor.from_pretrained(
            self._model_dir, local_files_only=True, trust_remote_code=True
        )

        self._model = ModelClass.from_pretrained(
            self._model_dir,
            dtype=torch.bfloat16,
            device_map='auto',
            local_files_only=True,
            trust_remote_code=True,
        )
        post_init(self._model)

        # LoRA 어댑터 적용 (있으면)
        adapter_path = latest_adapter()
        if adapter_path:
            try:
                from peft import PeftModel
                self._model = PeftModel.from_pretrained(
                    self._model, str(adapter_path), is_trainable=False
                )
                logger.info(f'[AutoRound] LoRA 어댑터 로드 완료: {adapter_path.name}')
            except Exception as e:
                logger.warning(f'[AutoRound] LoRA 어댑터 로드 실패: {e}')

        self._model.eval()
        mem_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info(f'[AutoRound] 로드 완료. GPU 메모리: {mem_gb:.1f}GB')

    def infer(
        self,
        image: Image.Image,
        user_context: str = '',
        audio: np.ndarray | None = None,
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
                    max_new_tokens=250,
                    do_sample=False,
                    pad_token_id=self._processor.tokenizer.eos_token_id,
                )

            input_len = inputs['input_ids'].shape[1]
            new_ids = output_ids[:, input_len:]
            raw_text = self._processor.tokenizer.decode(
                new_ids[0], skip_special_tokens=True
            )

            result = self._parse_response(raw_text)
            self._tom.add_assistant_turn(raw_text)
            return result

        except Exception as e:
            logger.error(f'[AutoRound] 추론 오류: {e}')
            return _make_fallback()

    def _parse_response(self, raw: str) -> HaruResponse:
        text = raw
        data = None

        m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        if data is None:
            for m in reversed(list(re.finditer(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', text, re.DOTALL))):
                try:
                    candidate = json.loads(m.group(1))
                    if 'action' in candidate:
                        data = candidate
                        break
                except json.JSONDecodeError:
                    continue

        if data is None:
            logger.warning(f'[AutoRound] JSON 없음: {text[:120]!r}')
            return _make_fallback()

        action = data.get('action', {})
        clamped = {}
        for joint, (lo, hi, default) in JOINT_LIMITS.items():
            v = action.get(joint, default)
            try:
                clamped[joint] = float(max(lo, min(hi, float(v))))
            except (ValueError, TypeError):
                clamped[joint] = float(default)

        return HaruResponse(
            speech        = str(data.get('speech', '')),
            emotion       = str(data.get('emotion', 'neutral')),
            expression_id = int(max(0, min(7, int(data.get('expression_id', 0))))),
            action        = clamped,
            duration      = float(max(0.5, min(10.0, float(data.get('duration', 2.5))))),
        )

    def reset_context(self):
        self._tom.reset()
