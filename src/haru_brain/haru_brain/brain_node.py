"""
HARU Brain Node — System 2 (Gemma 4 12B Unified)

Triple-System Phase 5.5: /haru_attention/event 구독으로 트리거.
60초 고정 타이머 제거 — attention_node가 트리거 전담.

구독:
  haru_attention/event     (String, JSON) — 트리거 + 상황 컨텍스트 (System 3)
  haru_audio/raw           (Float32MultiArray) — 오디오 데이터 (Gemma 4 멀티모달)

발행:
  haru_vla_raw             (String, JSON) → hitl_node 또는 action_node
  haru_expression          (Int32)
  haru_speech              (String)       — speech="" 가능 (Gemma 4 침묵 선택)

토픽 라우팅:
  - HITL 모드  : brain → haru_vla_raw → hitl_node → haru_system1_command → action_node
  - 직결 모드  : brain → haru_vla_raw → action_node

파라미터:
  audio_timeout  (float, 기본 5.0): 오디오 수신 후 이 시간 내에만 Gemma 4에 첨부
"""

import sys

_VENV_SITE = '/home/herobot/robot_brain_workspace/haru_vla_env/lib/python3.10/site-packages'
if _VENV_SITE not in sys.path:
    sys.path.insert(0, _VENV_SITE)

import io
import json
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Float32MultiArray
from sensor_msgs.msg import CompressedImage
from PIL import Image

from .gemma4_inference import Gemma4Inference


class HaruBrainNode(Node):
    def __init__(self):
        super().__init__('haru_brain_node')

        self.declare_parameter('audio_timeout', 5.0)
        self._audio_timeout = self.get_parameter('audio_timeout').get_parameter_value().double_value

        # ── 구독 ─────────────────────────────────────────────────────────────
        # System 3 → System 2 트리거 (상황 컨텍스트 포함)
        self.create_subscription(
            String, 'haru_attention/event', self._attention_cb, 10)
        # 카메라 (Gemma 4 멀티모달 입력용)
        self.create_subscription(
            CompressedImage, 'haru_vision/compressed', self._image_cb, 10)
        # 오디오 (Gemma 4 음성 입력용 — 트리거는 attention_node가 담당)
        self.create_subscription(
            Float32MultiArray, 'haru_audio/raw', self._audio_cb, 5)

        # ── 발행 ─────────────────────────────────────────────────────────────
        self.pub_raw    = self.create_publisher(String, 'haru_vla_raw',    10)
        self.pub_expr   = self.create_publisher(Int32,  'haru_expression', 10)
        self.pub_speech = self.create_publisher(String, 'haru_speech',     10)

        # ── 내부 상태 ─────────────────────────────────────────────────────────
        self._latest_frame: Image.Image | None = None
        self._frame_lock = threading.Lock()

        self._latest_audio: np.ndarray | None = None
        self._audio_recv_time: float = 0.0
        self._audio_lock = threading.Lock()

        # 추론 상태 — attention_node와 audio_cb 양쪽에서 접근하므로 lock 필수
        self._inferring = False
        self._infer_lock = threading.Lock()

        self._model_ready = False
        self._brain = None
        self._using_trtllm = False
        self._mode_tag = 'HF-bf16'

        # 모델 로드 (별도 스레드 — ROS2 spin 즉시 시작 가능)
        threading.Thread(target=self._load_model, daemon=True).start()

        self.get_logger().info(
            '[Brain] System 2 시작 — attention_node 이벤트 대기 중. 모델 로드 중...'
        )

    # ── 모델 로드 ────────────────────────────────────────────────────────────

    def _load_model(self):
        # ── 경로 1: TRT-LLM 서버 (고속, ~6~10s) ────────────────────────────
        try:
            from .gemma4_trtllm_inference import Gemma4TRTLLMInference
            trtllm = Gemma4TRTLLMInference()
            if trtllm.try_connect():
                trtllm.load()
                self._brain = trtllm
                self._using_trtllm = True
                self._mode_tag = 'TRT-LLM'
                self._model_ready = True
                self.get_logger().info(
                    '[Brain] ✅ TRT-LLM 서버 연결 성공 — 고속 추론 모드 (W4A16 INT4)'
                )
                return
            else:
                self.get_logger().info('[Brain] TRT-LLM 서버 없음 — HuggingFace 경로로 폴백')
        except Exception as e:
            self.get_logger().warn(f'[Brain] TRT-LLM 초기화 실패 ({e}) — HuggingFace 폴백')

        # ── 경로 2: auto_round W4A16 양자화 모델 (중속, ~10~20s) ────────────
        try:
            from .gemma4_autoround_inference import Gemma4AutoRoundInference, quantized_model_exists
            if quantized_model_exists():
                ar = Gemma4AutoRoundInference()
                ar.load()
                self._brain = ar
                self._using_trtllm = False
                self._mode_tag = 'AutoRound-W4A16'
                self._model_ready = True
                self.get_logger().info(
                    '[Brain] ✅ auto_round W4A16 양자화 모델 로드 완료 (~3-5× bf16 대비)'
                )
                return
            else:
                self.get_logger().info(
                    '[Brain] auto_round 양자화 모델 없음 — '
                    'scripts/quantize_gemma4_autoround.py 실행 후 사용 가능'
                )
        except Exception as e:
            self.get_logger().warn(f'[Brain] auto_round 로드 실패 ({e}) — bf16 폴백')

        # ── 경로 3: HuggingFace Transformers bf16 (기본 경로, GPU ~15~25s) ──
        try:
            self._brain = Gemma4Inference(load_in_4bit=False)
            self._brain.load()
            self._using_trtllm = False
            self._model_ready = True
            self.get_logger().info('[Brain] HuggingFace bf16 모델 준비 완료 (GPU 사용).')
        except Exception as e:
            self.get_logger().error(f'[Brain] 모델 로드 실패: {e}')

    # ── 콜백 ─────────────────────────────────────────────────────────────────

    def _image_cb(self, msg: CompressedImage):
        try:
            img = Image.open(io.BytesIO(bytes(msg.data))).convert('RGB')
            with self._frame_lock:
                self._latest_frame = img
        except Exception as e:
            self.get_logger().warn(f'이미지 디코딩 오류: {e}')

    def _audio_cb(self, msg: Float32MultiArray):
        """오디오 데이터 저장 — 트리거는 하지 않음. attention_node가 트리거 전담."""
        audio = np.array(msg.data, dtype=np.float32)
        with self._audio_lock:
            self._latest_audio    = audio
            self._audio_recv_time = time.time()
        dur = len(audio) / 16000
        self.get_logger().debug(f'[Brain] 오디오 수신: {dur:.2f}s (다음 추론 시 첨부 대기)')

    def _attention_cb(self, msg: String):
        """
        System 3 → System 2 트리거.
        상황 컨텍스트를 파싱해 Gemma 4 추론 시작.
        """
        try:
            event = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'[Brain] attention event 파싱 실패: {e}')
            return

        if not event.get('trigger', False):
            return

        context  = event.get('context', '')
        state    = event.get('state', '?')

        self.get_logger().info(f'[Brain] 트리거 수신 [{state}]: {context[:60]}')
        self._run_infer(context=context, source=state)

    # ── 추론 ─────────────────────────────────────────────────────────────────

    def _run_infer(self, context: str, source: str):
        if not self._model_ready:
            self.get_logger().debug('[Brain] 모델 미준비 — 트리거 무시')
            return

        with self._infer_lock:
            if self._inferring:
                self.get_logger().info(f'[Brain] 추론 중 — [{source}] 트리거 건너뜀')
                return

            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                self.get_logger().warn('[Brain] 카메라 프레임 없음 — 트리거 무시')
                return

            # VAD 트리거인 경우 최신 오디오 첨부 (audio_timeout 내에 수신된 것만)
            with self._audio_lock:
                audio = self._latest_audio
                if audio is not None:
                    age = time.time() - self._audio_recv_time
                    if age > self._audio_timeout:
                        audio = None
                    else:
                        self._latest_audio = None  # 한 번만 사용

            self._inferring = True

        audio_info = f'+오디오({len(audio)/16000:.1f}s)' if audio is not None else '비전만'
        self.get_logger().info(f'[Brain|{self._mode_tag}] [{source}] 추론 시작 ({audio_info})')
        threading.Thread(
            target=self._infer_worker,
            args=(frame, audio, context, source),
            daemon=True,
        ).start()


    def _infer_worker(
        self,
        frame: Image.Image,
        audio: np.ndarray | None,
        context: str,
        source: str,
    ):
        try:
            t0 = time.time()
            speech_published = [False]

            expr_published = [False]

            def _early_speech(text: str):
                """streaming: speech 필드 완성 즉시 TTS 발행 (체감 지연 단축)."""
                msg = String()
                msg.data = text
                self.pub_speech.publish(msg)
                speech_published[0] = True
                if text:
                    self.get_logger().info(f'[Brain] [STREAM] speech 조기 발행: {repr(text[:40])}')

            def _early_expr(eid: int):
                """streaming: expression_id 완성 즉시 표정 변경."""
                emsg = Int32()
                emsg.data = eid
                self.pub_expr.publish(emsg)
                expr_published[0] = True
                self.get_logger().info(f'[Brain] [STREAM] expression 조기 발행: {eid}')

            resp = self._brain.infer(frame, user_context=context, audio=audio,
                                     speech_ready_cb=_early_speech,
                                     expression_ready_cb=_early_expr)
            elapsed = time.time() - t0
            audio_tag = f' +audio({len(audio)/16000:.1f}s)' if audio is not None else ''
            speech_preview = f'"{resp.speech[:40]}"' if resp.speech else '(침묵)'
            self.get_logger().info(
                f'[Brain] [{source}] {elapsed:.1f}s{audio_tag} | '
                f'emotion={resp.emotion} | speech={speech_preview}'
            )
            self._publish(resp, context=context, source=source,
                          skip_speech=speech_published[0],
                          skip_expr=expr_published[0])
        except Exception as e:
            self.get_logger().error(f'[Brain] 추론 오류: {e}')
        finally:
            with self._infer_lock:
                self._inferring = False

    # ── 발행 ─────────────────────────────────────────────────────────────────

    def _publish(self, resp, context: str = '', source: str = '',
                 skip_speech: bool = False, skip_expr: bool = False):
        cmd = resp.to_command_dict()
        if context:
            cmd['attention_context'] = context
        if source:
            cmd['attention_source'] = source
        json_str = json.dumps(cmd, ensure_ascii=False)

        raw_msg = String()
        raw_msg.data = json_str
        self.pub_raw.publish(raw_msg)

        if not skip_expr:
            expr_msg = Int32()
            expr_msg.data = resp.expression_id
            self.pub_expr.publish(expr_msg)

        if not skip_speech:
            speech_msg = String()
            speech_msg.data = resp.speech  # "" 가능 — TTS 노드가 빈 문자열이면 침묵
            self.pub_speech.publish(speech_msg)

        act = cmd['action']
        arm_r = f'{act.get("r_arm_pitch",0):.0f}/{act.get("r_shoulder_roll",0):.0f}/{act.get("r_elbow_pitch",0):.0f}'
        arm_l = f'{act.get("l_arm_pitch",0):.0f}/{act.get("l_shoulder_roll",0):.0f}/{act.get("l_elbow_pitch",0):.0f}'
        self.get_logger().info(
            f'[Pub] expr={resp.expression_id} speech={"(침묵)" if not resp.speech else repr(resp.speech[:30])} '
            f'head=({act["head_tilt"]:.0f},{act["head_pan"]:.0f},{act["head_roll"]:.0f}) '
            f'r_arm={arm_r} l_arm={arm_l} wheel=({act.get("right_wheel",0):.0f},{act.get("left_wheel",0):.0f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = HaruBrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('종료')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
