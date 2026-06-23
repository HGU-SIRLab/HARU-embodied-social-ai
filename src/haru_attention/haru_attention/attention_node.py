"""
HARU Attention Node — System 3 (Triple-System, Phase 5.5)

항상 켜져 있는 경량 지각·주의 레이어.
Gemma 4를 언제 깨울지 결정하고, 상황 컨텍스트를 생성해 전달한다.

5-State FSM:
  EMPTY          → 방에 아무도 없음. brain 트리거 없음.
  APPEARED       → 사람 등장, appeared_wait_sec 동안 대기.
                   (상대방이 먼저 말 걸 기회를 줌)
  CONVERSING     → 활발한 대화 중. VAD 완료 시 즉시 트리거.
  PRESENT_SILENT → 사람이 있으나 대화 없음. idle_trigger_sec 주기 트리거.
  LONG_IDLE      → PRESENT_SILENT가 long_idle_sec 이상 지속. 매우 드문 체크인.

트리거 이벤트 (→ /haru_attention/event):
  APPEARED → PRESENT_SILENT 전환  : 즉시 트리거 (먼저 말 걸기)
  CONVERSING 상태에서 VAD 완료     : 즉시 트리거 (응답)
  PRESENT_SILENT 상태 주기         : idle_trigger_sec 마다
  LONG_IDLE 상태 주기              : long_idle_trigger_sec 마다

구독:
  /haru_vision/compressed  (CompressedImage) — 얼굴·모션 감지
  /haru_audio/vad          (Bool)            — 발화 감지

발행:
  /haru_attention/event    (String, JSON)    — 트리거 + 상황 컨텍스트

파라미터:
  appeared_wait_sec      (float, 기본 5.0)   — 등장 후 대기 시간
  face_lost_timeout      (float, 기본 3.0)   — 얼굴 소실 후 EMPTY 전환 대기
  conversation_timeout   (float, 기본 30.0)  — 침묵 후 대화 종료 판정
  idle_trigger_sec       (float, 기본 120.0) — PRESENT_SILENT 주기 트리거
  long_idle_sec          (float, 기본 600.0) — LONG_IDLE 전환 기준 (10분)
  long_idle_trigger_sec  (float, 기본 300.0) — LONG_IDLE 트리거 주기 (5분)
  face_min_size          (int,   기본 80)    — 얼굴로 인정할 최소 픽셀 크기
  motion_threshold       (float, 기본 8.0)   — 모션 감지 임계값 (프레임 평균 차이)
"""

import io
import json
import time
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from sensor_msgs.msg import CompressedImage

# Haar Cascade 얼굴 감지기 (CPU, ~5ms/frame)
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# FSM 상태 상수
_EMPTY          = 'EMPTY'
_APPEARED       = 'APPEARED'
_CONVERSING     = 'CONVERSING'
_PRESENT_SILENT = 'PRESENT_SILENT'
_LONG_IDLE      = 'LONG_IDLE'


def _detect_face(bgr: np.ndarray, min_size: int) -> bool:
    """BGR 이미지에서 얼굴 존재 여부 반환 (~5ms, CPU)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(min_size, min_size),
    )
    return len(faces) > 0


def _motion_score(prev: np.ndarray | None, curr: np.ndarray) -> float:
    """이전 프레임과 현재 프레임의 평균 픽셀 차이 반환 (~1ms)."""
    if prev is None:
        return 0.0
    diff = cv2.absdiff(prev, curr)
    return float(diff.mean())


class AttentionFSM:
    """
    상황 상태 머신.
    외부에서 face_present / vad_active 이벤트를 주입하면
    현재 상태와 트리거 여부를 판단한다.
    """

    def __init__(
        self,
        appeared_wait_sec: float,
        face_lost_timeout: float,
        conversation_timeout: float,
        idle_trigger_sec: float,
        long_idle_sec: float,
        long_idle_trigger_sec: float,
    ):
        self._appeared_wait      = appeared_wait_sec
        self._face_lost_timeout  = face_lost_timeout
        self._conv_timeout       = conversation_timeout
        self._idle_trigger_sec   = idle_trigger_sec
        self._long_idle_sec      = long_idle_sec
        self._long_idle_trigger  = long_idle_trigger_sec

        self._state              = _EMPTY
        self._face_last_seen: float = 0.0   # 마지막으로 얼굴 감지된 시각
        self._state_entered: float  = 0.0   # 현재 상태 진입 시각
        self._last_speech: float    = 0.0   # 마지막 발화 완료 시각
        self._last_trigger: float   = 0.0   # 마지막 brain 트리거 시각
        self._lock = threading.Lock()

    # ── 외부 이벤트 주입 ──────────────────────────────────────────────────────

    def on_frame(self, face_present: bool) -> dict | None:
        """
        새 프레임 처리 결과를 받아 FSM 갱신.
        brain 트리거가 필요하면 이벤트 dict 반환, 아니면 None.
        """
        with self._lock:
            now = time.time()
            if face_present:
                self._face_last_seen = now
            return self._update(now)

    def on_vad_complete(self) -> dict | None:
        """발화 완료(audio_node flush) 시 호출. 즉시 트리거 여부 반환.

        _update()를 경유하지 않고 직접 처리한다.
        이유: _update()는 face_present를 체크하여 소리만 들리고 얼굴이
        감지되지 않는 상황(off-camera 발화 등)에서 방금 전환한 CONVERSING을
        즉시 EMPTY로 되돌리는 레이스를 유발하기 때문.
        """
        with self._lock:
            now = time.time()
            self._last_speech = now
            # 소리가 들렸으므로 사람이 있다고 간주 — face_last_seen 갱신
            self._face_last_seen = now

            if self._state == _EMPTY:
                self._transition(_CONVERSING, now)
                return self._make_event(
                    state='CONVERSING',
                    context='사용자가 말을 걸었음. 자연스럽게 응답하세요.',
                    now=now,
                )
            if self._state == _APPEARED:
                self._transition(_CONVERSING, now)
                return self._make_event(
                    state='CONVERSING',
                    context='사용자가 먼저 말을 걸었음. 자연스럽게 응답하세요.',
                    now=now,
                )
            if self._state in (_PRESENT_SILENT, _LONG_IDLE):
                self._transition(_CONVERSING, now)
                return self._make_event(
                    state='CONVERSING',
                    context='한동안 조용하던 사용자가 말을 걸었음. 반갑게 응답하세요.',
                    now=now,
                )
            if self._state == _CONVERSING:
                return self._make_event(
                    state='CONVERSING',
                    context='사용자가 방금 말을 마쳤음. 자연스럽게 응답하세요.',
                    now=now,
                )
            return None

    # ── 내부 FSM 로직 ─────────────────────────────────────────────────────────

    def _update(self, now: float) -> dict | None:
        face_present = (now - self._face_last_seen) < self._face_lost_timeout
        state_dur    = now - self._state_entered

        # ── EMPTY ────────────────────────────────────────────────────────────
        if self._state == _EMPTY:
            if face_present:
                self._transition(_APPEARED, now)
            return None  # EMPTY에서는 트리거 없음

        # ── APPEARED ─────────────────────────────────────────────────────────
        if self._state == _APPEARED:
            if not face_present:
                self._transition(_EMPTY, now)
                return None
            if state_dur >= self._appeared_wait:
                # 기다렸는데 말 없음 → PRESENT_SILENT 전환 + 즉시 트리거
                self._transition(_PRESENT_SILENT, now)
                secs = int(state_dur)
                return self._make_event(
                    state='APPEARED_TO_SILENT',
                    context=f'사람이 {secs}초 전에 등장했고 아직 말을 걸지 않음. '
                             '먼저 인사할지 판단하세요.',
                    now=now,
                )
            return None

        # ── CONVERSING ───────────────────────────────────────────────────────
        if self._state == _CONVERSING:
            if not face_present:
                self._transition(_EMPTY, now)
                return None
            silence_dur = now - self._last_speech
            if silence_dur > self._conv_timeout:
                self._transition(_PRESENT_SILENT, now)
                return None
            return None

        # ── PRESENT_SILENT ───────────────────────────────────────────────────
        if self._state == _PRESENT_SILENT:
            if not face_present:
                self._transition(_EMPTY, now)
                return None
            # LONG_IDLE 전환 체크
            if state_dur >= self._long_idle_sec:
                self._transition(_LONG_IDLE, now)
                return None
            # 주기적 체크인
            since_trigger = now - self._last_trigger
            if self._last_trigger == 0.0 or since_trigger >= self._idle_trigger_sec:
                mins = int(state_dur / 60)
                secs = int(state_dur % 60)
                dur_str = f'{mins}분 {secs}초' if mins > 0 else f'{secs}초'
                return self._make_event(
                    state='PRESENT_SILENT',
                    context=f'사람이 {dur_str}째 있으나 상호작용이 없음. '
                             '방해가 되지 않게 조심스럽게 말을 걸지 판단하세요.',
                    now=now,
                )
            return None

        # ── LONG_IDLE ────────────────────────────────────────────────────────
        if self._state == _LONG_IDLE:
            if not face_present:
                self._transition(_EMPTY, now)
                return None
            since_trigger = now - self._last_trigger
            if self._last_trigger == 0.0 or since_trigger >= self._long_idle_trigger:
                total_mins = int((now - self._state_entered + self._long_idle_sec) / 60)
                return self._make_event(
                    state='LONG_IDLE',
                    context=f'사람이 약 {total_mins}분째 있으나 대화가 없음. '
                             '매우 조심스럽게 안부를 물을지 판단하세요.',
                    now=now,
                )
            return None

        return None

    def _transition(self, new_state: str, now: float):
        self._state         = new_state
        self._state_entered = now

    def _make_event(self, state: str, context: str, now: float) -> dict:
        self._last_trigger = now
        last_speech_ago = (now - self._last_speech) if self._last_speech > 0 else None
        return {
            'trigger':           True,
            'state':             state,
            'context':           context,
            'fsm_state':         self._state,
            'face_present':      True,
            'face_duration_sec': round(now - self._state_entered, 1),
            'last_speech_ago_sec': round(last_speech_ago, 1) if last_speech_ago else None,
        }

    @property
    def state(self) -> str:
        return self._state


class HaruAttentionNode(Node):
    def __init__(self):
        super().__init__('haru_attention_node')

        # ── 파라미터 ─────────────────────────────────────────────────────────
        self.declare_parameter('appeared_wait_sec',    5.0)
        self.declare_parameter('face_lost_timeout',    3.0)
        self.declare_parameter('conversation_timeout', 30.0)
        self.declare_parameter('idle_trigger_sec',     120.0)
        self.declare_parameter('long_idle_sec',        600.0)
        self.declare_parameter('long_idle_trigger_sec',300.0)
        self.declare_parameter('face_min_size',        80)
        self.declare_parameter('motion_threshold',     8.0)

        p = self.get_parameter
        self._face_min_size     = p('face_min_size').value
        self._motion_threshold  = p('motion_threshold').value

        self._fsm = AttentionFSM(
            appeared_wait_sec    = p('appeared_wait_sec').value,
            face_lost_timeout    = p('face_lost_timeout').value,
            conversation_timeout = p('conversation_timeout').value,
            idle_trigger_sec     = p('idle_trigger_sec').value,
            long_idle_sec        = p('long_idle_sec').value,
            long_idle_trigger_sec= p('long_idle_trigger_sec').value,
        )

        # ── 구독 ─────────────────────────────────────────────────────────────
        self.create_subscription(
            CompressedImage, 'haru_vision/compressed', self._image_cb, 5)
        self.create_subscription(
            Bool, 'haru_audio/vad', self._vad_cb, 10)

        # ── 발행 ─────────────────────────────────────────────────────────────
        self.pub_event = self.create_publisher(String, 'haru_attention/event', 10)

        # ── 내부 상태 ─────────────────────────────────────────────────────────
        self._prev_gray: np.ndarray | None = None
        self._vad_was_active = False   # VAD rising/falling edge 감지용
        self._frame_lock = threading.Lock()

        self.get_logger().info(
            f'[Attention] System 3 시작 — '
            f'appeared_wait={p("appeared_wait_sec").value}s, '
            f'idle_trigger={p("idle_trigger_sec").value}s'
        )

    # ── 이미지 콜백 ───────────────────────────────────────────────────────────

    def _image_cb(self, msg: CompressedImage):
        try:
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                return
        except Exception as e:
            self.get_logger().warn(f'[Attention] 이미지 디코딩 실패: {e}')
            return

        # vision_node: hconcat([RealSense_face | C270_body]) → 896×448
        # 얼굴 감지는 RealSense(왼쪽 절반)만 사용 — 몸통 카메라는 제외
        h, w = bgr.shape[:2]
        face_roi = bgr[:, : w // 2]   # 왼쪽 절반 = RealSense 얼굴 카메라

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 모션 감지 (전체 이미지 — 몸통 움직임도 포함해 더 민감)
        motion = _motion_score(self._prev_gray, gray)
        self._prev_gray = gray

        # 얼굴 감지: RealSense 영역만
        face_present = _detect_face(face_roi, self._face_min_size)

        prev_state = self._fsm.state
        event = self._fsm.on_frame(face_present)

        # 상태 전환 로그
        if self._fsm.state != prev_state:
            self.get_logger().info(
                f'[Attention] FSM: {prev_state} → {self._fsm.state} '
                f'(face={face_present}, motion={motion:.1f})'
            )

        if event:
            self._publish_event(event)

    # ── VAD 콜백 ──────────────────────────────────────────────────────────────

    def _vad_cb(self, msg: Bool):
        """
        VAD Bool 토픽은 발화 중(True) / 침묵(False)을 계속 발행.
        audio_node는 발화가 끝날 때 /haru_audio/raw를 발행하고
        이 Bool은 실시간 상태를 나타냄.
        우리는 True→False 하강 엣지(발화 완료)를 감지해 FSM에 전달.
        """
        is_speaking = bool(msg.data)

        if self._vad_was_active and not is_speaking:
            # 발화 완료 (하강 엣지)
            event = self._fsm.on_vad_complete()
            if event:
                self.get_logger().info(
                    f'[Attention] VAD 완료 트리거 — state={self._fsm.state}'
                )
                self._publish_event(event)

        self._vad_was_active = is_speaking

    # ── 발행 ──────────────────────────────────────────────────────────────────

    def _publish_event(self, event: dict):
        msg = String()
        msg.data = json.dumps(event, ensure_ascii=False)
        self.pub_event.publish(msg)
        self.get_logger().info(
            f'[Attention] → brain 트리거: [{event["state"]}] {event["context"][:50]}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = HaruAttentionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('[Attention] 종료')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
