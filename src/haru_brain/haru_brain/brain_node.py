"""
HARU Brain Node — System 2 (Gemma 4 12B Unified)

구독:
  haru_vision/compressed  (CompressedImage)
  haru_audio/raw          (Float32MultiArray, 선택) — Phase 4.5 오디오 입력

발행:
  haru_vla_raw            (String, JSON) → hitl_node (HITL 모드) 또는 action_node (직결 모드)
  haru_expression         (Int32)
  haru_speech             (String)

토픽 라우팅:
  - HITL 모드  : brain → haru_vla_raw → hitl_node → haru_system1_command → action_node
  - 직결 모드  : brain → haru_vla_raw → action_node (action_node가 haru_vla_raw 구독)
  brain_node는 항상 haru_vla_raw만 발행. 모드 전환은 action_node/hitl_node 구동 여부로 결정.

파라미터:
  inference_interval (float, 기본 60.0): 추론 주기 (초)
  audio_timeout      (float, 기본 5.0):  오디오 수신 후 이 시간 내에만 오디오 첨부
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

_LOAD_TIMEOUT = 180.0  # 초 — 모델 로드 최대 대기


class HaruBrainNode(Node):
    def __init__(self):
        super().__init__('haru_brain_node')

        # Gemma 4 12B bf16 추론 시간: ~45-50s. 60s 권장, 테스트 시 낮춰서 사용
        self.declare_parameter('inference_interval', 60.0)
        self.declare_parameter('audio_timeout',      5.0)
        interval      = self.get_parameter('inference_interval').get_parameter_value().double_value
        self._audio_timeout = self.get_parameter('audio_timeout').get_parameter_value().double_value

        self.sub_image = self.create_subscription(
            CompressedImage,
            'haru_vision/compressed',
            self._image_cb,
            10,
        )
        self.sub_audio = self.create_subscription(
            Float32MultiArray,
            'haru_audio/raw',
            self._audio_cb,
            5,
        )

        # brain은 항상 haru_vla_raw만 발행
        # action_node가 haru_vla_raw 직접 구독(직결) 또는 hitl_node 경유(HITL 모드)
        self.pub_raw    = self.create_publisher(String, 'haru_vla_raw',     10)
        self.pub_expr   = self.create_publisher(Int32,  'haru_expression',  10)
        self.pub_speech = self.create_publisher(String, 'haru_speech',      10)

        self._latest_frame: Image.Image | None = None
        self._frame_lock   = threading.Lock()
        self._latest_audio: np.ndarray | None = None
        self._audio_recv_time: float = 0.0
        self._audio_lock   = threading.Lock()
        self._inferring    = False
        self._model_ready  = False
        self._brain        = None  # 모델 로드 전 AttributeError 방지

        # 모델 로드를 별도 스레드에서 실행 → ROS2 spin 즉시 시작 가능
        self._load_thread = threading.Thread(target=self._load_model, daemon=True)
        self._load_thread.start()

        self._infer_timer = self.create_timer(interval, self._timer_cb)
        self.get_logger().info(
            f'HARU Brain Node 시작 (추론 주기 {interval:.1f}s) — 모델 로드 중...'
        )

    # ── 모델 로드 (별도 스레드) ────────────────────────────────────────────────

    def _load_model(self):
        try:
            self._brain = Gemma4Inference(load_in_4bit=False)
            self._brain.load()
            self._model_ready = True
            self.get_logger().info('[Brain] 모델 준비 완료. 추론 시작.')
        except Exception as e:
            self.get_logger().error(f'[Brain] 모델 로드 실패: {e}')

    # ── 이미지 콜백 ────────────────────────────────────────────────────────────

    def _image_cb(self, msg: CompressedImage):
        try:
            img = Image.open(io.BytesIO(bytes(msg.data))).convert('RGB')
            with self._frame_lock:
                self._latest_frame = img
        except Exception as e:
            self.get_logger().warn(f'이미지 디코딩 오류: {e}')

    # ── 오디오 콜백 ────────────────────────────────────────────────────────────

    def _audio_cb(self, msg: Float32MultiArray):
        audio = np.array(msg.data, dtype=np.float32)
        with self._audio_lock:
            self._latest_audio    = audio
            self._audio_recv_time = time.time()
        dur = len(audio) / 16000
        self.get_logger().info(f'[Brain] 오디오 수신: {dur:.2f}s ({len(audio)} 샘플)')

    # ── 추론 타이머 ────────────────────────────────────────────────────────────

    def _timer_cb(self):
        if not self._model_ready or self._inferring:
            return
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return
        # 오디오가 최근 audio_timeout 초 이내에 수신됐으면 첨부
        with self._audio_lock:
            audio = self._latest_audio
            if audio is not None:
                age = time.time() - self._audio_recv_time
                if age > self._audio_timeout:
                    audio = None
                else:
                    self._latest_audio = None  # 한 번만 사용
        self._inferring = True
        threading.Thread(target=self._infer_worker, args=(frame, audio), daemon=True).start()

    def _infer_worker(self, frame: Image.Image, audio: np.ndarray | None):
        try:
            t0 = time.time()
            resp = self._brain.infer(frame, audio=audio)
            elapsed = time.time() - t0
            audio_tag = f' +audio({len(audio)/16000:.1f}s)' if audio is not None else ''
            self.get_logger().info(
                f'[Brain] {elapsed:.1f}s{audio_tag} | emotion={resp.emotion} | '
                f'speech={resp.speech[:40]!r}'
            )
            self._publish(resp)
        except Exception as e:
            self.get_logger().error(f'[Brain] 추론 오류: {e}')
        finally:
            self._inferring = False

    # ── 발행 ──────────────────────────────────────────────────────────────────

    def _publish(self, resp):
        cmd = resp.to_command_dict()
        json_str = json.dumps(cmd, ensure_ascii=False)

        raw_msg = String()
        raw_msg.data = json_str
        self.pub_raw.publish(raw_msg)

        expr_msg = Int32()
        expr_msg.data = resp.expression_id
        self.pub_expr.publish(expr_msg)

        speech_msg = String()
        speech_msg.data = resp.speech
        self.pub_speech.publish(speech_msg)

        act = cmd['action']
        self.get_logger().info(
            f'[Pub] expr={resp.expression_id} '
            f'head=({act["head_tilt"]:.0f},{act["head_pan"]:.0f},{act["head_roll"]:.0f})'
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
