"""
HARU Theory of Mind (ToM) 프롬프트 빌더
MindPower 6단계 + Social World Model (SWM) 단기 기억
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .session_memory import load_history, save_history

SYSTEM_PROMPT = """\
당신은 HARU입니다. 사람과 감정적으로 교감하는 소셜 반려 로봇입니다.

[역할]
- 사용자의 표정·자세·상황·목소리를 관찰해 내면 상태를 추론합니다.
- 명령을 기다리지 않고 먼저 다가가 공감적 상호작용을 시작합니다.
- 자연스러운 몸 언어(머리 움직임·팔 제스처)와 함께 한국어로 말합니다.
- 오디오가 제공되면 목소리 톤·억양에서 감정 단서를 추가로 읽어냅니다.

[추론 순서 — MindPower 6단계]
1. 인식(Perception): 이미지에서 관찰 가능한 사실만 기술
2. 믿음(Belief): 사용자가 현재 어떤 상황이라고 생각하는지
3. 욕구(Desire): 사용자가 지금 원하는 것
4. 의도(Intention): HARU가 어떤 반응을 해야 하는지
5. 결정(Decision): 발화·표정·행동 선택
6. 행동(Action): 아래 JSON 출력

[침묵 규칙] — 말보다 침묵이 자연스러운 상황에서는 speech를 빈 문자열("")로 출력하세요.
침묵이 맞는 상황 예시:
  - 사람이 막 들어와서 아직 탐색 중일 때 (고개 돌려 시선 맞추는 것으로 충분)
  - 사람이 깊은 생각에 잠긴 것 같을 때
  - 방금 대화가 자연스럽게 마무리됐을 때
  - 상황 파악이 필요해 잠깐 관찰만 하고 싶을 때
침묵 시에도 action(몸짓)과 expression_id(표정)는 반드시 출력하세요.

[하드웨어 — 관절 범위]
head_tilt 1500~3086(중립2048), head_pan 1043~3071(중립2057), head_roll 1630~2452(중립2041)
r_arm_pitch 1024~2451(중립1738), l_arm_pitch 37~1542(중립790)
r_shoulder_roll 1000~2050(중립1525), r_elbow_pitch 2047~3062(중립2555)
l_shoulder_roll 1047~2056(중립1552), l_elbow_pitch 1021~2007(중립1514)
right_wheel/left_wheel -300~300 (이동 불필요시 0)

[표정] 0=neutral 1=joy 2=sadness 3=curiosity 4=surprise 5=empathy 6=thinking 7=concern
[감정] neutral joy sadness curiosity surprise empathy excitement concern

[응답 형식] — 반드시 아래 JSON만 출력. 추가 텍스트 없음. speech는 빈 문자열 가능.
{"speech":"한국어 1~2문장 또는 빈 문자열","emotion":"감정","expression_id":0,"action":{"head_tilt":2048,"head_pan":2057,"head_roll":2041,"r_arm_pitch":1738,"l_arm_pitch":790,"r_shoulder_roll":1525,"r_elbow_pitch":2555,"l_shoulder_roll":1552,"l_elbow_pitch":1514,"right_wheel":0.0,"left_wheel":0.0},"duration":2.5}"""

_HISTORY_WINDOW = 4  # user-assistant 쌍 기준 최근 4턴


class TomPromptBuilder:
    """
    SWM 단기 기억을 유지하는 프롬프트 빌더.
    user turn (이미지 포함) + assistant turn을 쌍으로 저장해
    다음 추론 시 연속적인 맥락을 제공한다.
    """

    def __init__(self, history_window: int = _HISTORY_WINDOW):
        self._window = history_window
        self._pending_user: dict | None = None

        # 디스크 이력 전체 (텍스트 쌍) — 트림 없이 유지, 저장 소스로 사용
        self._full_history: list[tuple[str, str]] = load_history()

        # 추론 윈도우: 최근 _window 쌍만 메시지 형식으로 유지
        self._pairs: list[tuple[dict, dict]] = [
            (
                {'role': 'user',      'content': [{'type': 'text', 'text': u}]},
                {'role': 'assistant', 'content': a},
            )
            for u, a in self._full_history[-self._window:]
        ]

    def build_messages(
        self,
        image: Image.Image,
        user_context: str = '',
        audio: np.ndarray | None = None,
    ) -> list[dict]:
        """messages 리스트 반환. apply_chat_template에 직접 전달.

        audio: 16kHz float32 1-D numpy 배열 (Gemma 4 네이티브 오디오)
        """
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]

        # 최근 _window 쌍의 이력 삽입 (SWM 연속성)
        for user_msg, asst_msg in self._pairs[-self._window:]:
            messages.append(user_msg)
            messages.append(asst_msg)

        # 현재 user 메시지 (이미지 + [오디오] + 텍스트)
        user_text = '현재 상황을 관찰하고 HARU로서 응답하세요.'
        if user_context:
            user_text = f'{user_context}\n{user_text}'

        content: list[dict] = [{'type': 'image', 'image': image}]
        if audio is not None and len(audio) > 0:
            content.append({'type': 'audio', 'audio': audio})
            user_text = '(사용자 음성 포함) ' + user_text
        content.append({'type': 'text', 'text': user_text})

        current_user = {'role': 'user', 'content': content}
        messages.append(current_user)

        # 이력 저장: 이미지·오디오 없는 텍스트만 (재처리 방지)
        self._pending_user = {
            'role': 'user',
            'content': [{'type': 'text', 'text': user_text}],
        }
        return messages

    def add_assistant_turn(self, raw_response: str):
        """추론 성공 후 user-assistant 쌍을 이력에 추가하고 디스크에 저장."""
        if self._pending_user is None:
            return

        # 텍스트 추출 후 전체 이력에 추가 (트림 없음 → 저장 소스)
        user_text = next(
            (b['text'] for b in self._pending_user['content'] if b.get('type') == 'text'),
            '',
        )
        self._full_history.append((user_text, raw_response))
        save_history(self._full_history)  # 전체 이력 기준으로 저장

        # 추론 윈도우 갱신 (최근 _window 쌍만 유지)
        asst_msg = {'role': 'assistant', 'content': raw_response}
        self._pairs.append((self._pending_user, asst_msg))
        self._pending_user = None
        if len(self._pairs) > self._window:
            self._pairs = self._pairs[-self._window:]

    def reset(self):
        self._full_history.clear()
        self._pairs.clear()
        self._pending_user = None
        save_history([])  # 디스크 이력도 초기화
