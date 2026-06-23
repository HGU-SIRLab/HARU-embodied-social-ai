"""
Gemma 4 12B TRT-LLM 추론 엔진 (HARU System 2 — 고속 경로)

TRT-LLM OpenAI 호환 서버(포트 8000)에 HTTP로 연결해 추론.
인터페이스는 Gemma4Inference와 동일 → brain_node가 투명하게 교체 가능.

HF bf16 대비 예상 속도: 5~8× (W4A16 INT4, SM87 Jetson AGX Orin 기준)
   HF bf16 ~45s → TRT-LLM W4A16 ~6~10s

연결 실패 시 try_connect()가 False를 반환 → brain_node가 HF 경로로 폴백.
"""

from __future__ import annotations

import base64
import copy
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .tom_prompt import TomPromptBuilder, SYSTEM_PROMPT
from .adapter_manager import latest_adapter
from .gemma4_inference import HaruResponse, JOINT_LIMITS, _NEUTRAL_ACTION, _make_fallback

logger = logging.getLogger(__name__)

_TRTLLM_SERVER_URL = 'http://localhost:8000'
_CONNECT_TIMEOUT   = 5.0   # 서버 연결 확인 타임아웃 (초)


def _image_to_b64(image: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


class Gemma4TRTLLMInference:
    """
    TRT-LLM 서버 기반 Gemma 4 12B 추론.
    Gemma4Inference와 동일한 public interface:
      load() / infer(image, user_context, audio) / reset_context()
    """

    def __init__(
        self,
        server_url: str = _TRTLLM_SERVER_URL,
        model_id:   str = 'gemma-4-12B-it',
    ):
        self._server_url = server_url.rstrip('/')
        self._model_id   = model_id
        self._client     = None
        self._tom        = TomPromptBuilder()
        self._model_name: str | None = None   # 서버에서 조회한 실제 모델 이름

    # ── 연결 & 로드 ───────────────────────────────────────────────────

    def try_connect(self) -> bool:
        """서버 응답 여부만 확인. 연결 성공 시 True."""
        try:
            import httpx
            r = httpx.get(f'{self._server_url}/v1/models', timeout=_CONNECT_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                models = [m['id'] for m in data.get('data', [])]
                if models:
                    self._model_name = models[0]
                    logger.info(f'[TRT-LLM] 서버 연결 성공. 모델: {self._model_name}')
                return True
        except Exception as e:
            logger.debug(f'[TRT-LLM] 연결 실패: {e}')
        return False

    def load(self):
        """OpenAI 클라이언트 초기화. 모델 파일 로드 없음 (서버가 보유)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                'openai 패키지 필요: pip install openai'
            )

        self._client = OpenAI(
            base_url=f'{self._server_url}/v1',
            api_key='EMPTY',
            timeout=120.0,
        )

        # 서버 모델 목록에서 실제 모델 이름 확인
        if self._model_name is None:
            try:
                models = self._client.models.list()
                if models.data:
                    self._model_name = models.data[0].id
            except Exception:
                self._model_name = self._model_id

        logger.info(
            f'[TRT-LLM] 준비 완료 — 서버: {self._server_url}, '
            f'모델: {self._model_name}'
        )

    # ── 추론 ─────────────────────────────────────────────────────────

    def infer(
        self,
        image: Image.Image,
        user_context: str = '',
        audio: np.ndarray | None = None,
    ) -> HaruResponse:
        if self._client is None:
            raise RuntimeError('load()를 먼저 호출하세요.')

        # HF 형식 메시지 생성 (SWM 이력 포함) → OpenAI 형식으로 변환
        hf_messages = self._tom.build_messages(image, user_context, audio=None)
        oai_messages = self._to_openai_format(hf_messages, image)

        t0 = time.time()
        try:
            resp = self._client.chat.completions.create(
                model=self._model_name or self._model_id,
                messages=oai_messages,
                max_tokens=300,
                temperature=0.0,
                stream=False,
            )
            raw_text = resp.choices[0].message.content or ''
            elapsed = time.time() - t0
            logger.info(
                f'[TRT-LLM] 추론 완료: {elapsed:.1f}s | '
                f'토큰: {resp.usage.completion_tokens if resp.usage else "?"}'
            )
            result = self._parse_response(raw_text)
            self._tom.add_assistant_turn(raw_text)
            return result

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f'[TRT-LLM] 추론 오류 ({elapsed:.1f}s): {e}')
            return _make_fallback()

    # ── 포맷 변환 ─────────────────────────────────────────────────────

    def _to_openai_format(
        self, hf_messages: list[dict], image: Image.Image
    ) -> list[dict]:
        """
        HuggingFace 메시지 형식 → OpenAI Vision API 형식.
        - {'type':'image', 'image': PIL}
          → {'type':'image_url', 'image_url':{'url':'data:image/jpeg;base64,...'}}
        - {'type':'audio', ...} 항목 제거 (TRT-LLM 12B 비지원)
        - 나머지는 그대로
        """
        image_b64 = _image_to_b64(image)
        oai_msgs: list[dict] = []

        for msg in hf_messages:
            content = msg['content']
            if isinstance(content, str):
                oai_msgs.append(msg)
                continue

            new_content: list[dict] = []
            for item in content:
                t = item.get('type', '')
                if t == 'image':
                    new_content.append({
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{image_b64}',
                        },
                    })
                elif t == 'audio':
                    pass  # 12B 모델 미지원 — 제거
                else:
                    new_content.append(item)

            if new_content:
                oai_msgs.append({'role': msg['role'], 'content': new_content})

        return oai_msgs

    # ── JSON 파싱 (Gemma4Inference와 동일) ────────────────────────────

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
            candidates = list(
                re.finditer(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', text, re.DOTALL)
            )
            for m in reversed(candidates):
                try:
                    candidate = json.loads(m.group(1))
                    if 'action' in candidate:
                        data = candidate
                        break
                except json.JSONDecodeError:
                    continue

        if data is None:
            logger.warning(f'[TRT-LLM] JSON 없음: {text[:120]!r}')
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
            speech       = str(data.get('speech', '')),
            emotion      = str(data.get('emotion', 'neutral')),
            expression_id= int(max(0, min(7, int(data.get('expression_id', 0))))),
            action       = clamped,
            duration     = float(max(0.5, min(10.0, float(data.get('duration', 2.5))))),
        )

    # ── 기타 ─────────────────────────────────────────────────────────

    def reset_context(self):
        self._tom.reset()
