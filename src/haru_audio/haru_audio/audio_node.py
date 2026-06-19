"""
HARU Audio Node — Phase 4.5

마이크(C270 USB Audio)에서 16kHz mono float32 오디오를 캡처하고
에너지 기반 VAD로 발화 구간을 감지하여 /haru_audio/raw 토픽에 발행.

아키텍처:
  sounddevice callback → raw_queue → 처리 스레드 (리샘플링 + VAD + 발행)
  callback은 raw 데이터 큐 투입만 수행하여 input overflow 방지.

구독: 없음
발행:
  /haru_audio/raw  (Float32MultiArray) — 발화 버퍼 (16kHz float32 PCM)
  /haru_audio/vad  (Bool)             — 현재 발화 중 여부

파라미터:
  device_name     (str,   기본 'C270')  — sounddevice 장치 이름 부분 일치
  vad_threshold   (float, 기본 0.01)   — RMS 에너지 발화 임계값
  speech_timeout  (float, 기본 1.0)    — 침묵 후 발화 종료 판정 (초)
  max_duration    (float, 기본 10.0)   — 단일 발화 최대 길이 (초)
  min_duration    (float, 기본 0.3)    — 발화로 인정되는 최소 길이 (초)
"""

import sys

_VENV_SITE = '/home/herobot/robot_brain_workspace/haru_vla_env/lib/python3.10/site-packages'
if _VENV_SITE not in sys.path:
    sys.path.insert(0, _VENV_SITE)

import queue
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension

try:
    import sounddevice as sd
    _HAS_SD = True
except ImportError:
    _HAS_SD = False

try:
    from scipy.signal import resample_poly
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_TARGET_SR = 16000  # Gemma 4 네이티브 오디오 샘플레이트
_CHUNK_MS  = 100    # 캡처 청크 크기 (ms)


def _find_device(name_hint: str) -> tuple[int | None, int]:
    """장치 이름에 name_hint가 포함된 입력 장치의 (인덱스, 네이티브 샘플레이트) 반환."""
    for i, dev in enumerate(sd.query_devices()):
        if name_hint.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
            return i, int(dev['default_samplerate'])
    return None, _TARGET_SR


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int = _TARGET_SR) -> np.ndarray:
    """48kHz → 16kHz 등 정수비 다운샘플링 (scipy 기반, 폴백: 정수 슬라이싱)."""
    if src_sr == dst_sr:
        return audio
    if _HAS_SCIPY:
        from math import gcd
        g = gcd(dst_sr, src_sr)
        return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)
    ratio = src_sr // dst_sr
    return audio[::ratio].astype(np.float32)


class HaruAudioNode(Node):
    def __init__(self):
        super().__init__('haru_audio_node')

        self.declare_parameter('device_name',    'C270')
        self.declare_parameter('vad_threshold',  0.01)
        self.declare_parameter('speech_timeout', 1.0)
        self.declare_parameter('max_duration',   10.0)
        self.declare_parameter('min_duration',   0.3)

        self._device_name    = self.get_parameter('device_name').value
        self._vad_thresh     = self.get_parameter('vad_threshold').value
        self._speech_timeout = self.get_parameter('speech_timeout').value
        self._max_dur        = self.get_parameter('max_duration').value
        self._min_dur        = self.get_parameter('min_duration').value

        self._capture_sr: int = _TARGET_SR

        self.pub_raw = self.create_publisher(Float32MultiArray, 'haru_audio/raw', 5)
        self.pub_vad = self.create_publisher(Bool, 'haru_audio/vad', 10)

        # destroy_node()에서 항상 접근하므로 _HAS_SD 체크 전에 초기화
        self._stream = None

        if not _HAS_SD:
            self.get_logger().error('sounddevice 미설치. audio_node 비활성화.')
            return

        # callback → 처리 스레드 간 통신용 큐
        self._raw_queue: queue.Queue[tuple[np.ndarray, str | None]] = queue.Queue(maxsize=50)

        # VAD 상태 (처리 스레드에서만 접근)
        self._buffer: list[np.ndarray] = []
        self._speaking = False
        self._silence_chunks = 0

        threading.Thread(target=self._start_stream, daemon=True).start()
        threading.Thread(target=self._process_loop,  daemon=True).start()

    # ── 스트림 초기화 ──────────────────────────────────────────────────────────

    def _start_stream(self):
        device_idx, native_sr = _find_device(self._device_name)
        if device_idx is None:
            self.get_logger().warn(
                f"'{self._device_name}' 장치 없음. 기본 입력 장치 사용."
            )
            native_sr = int(sd.query_devices(kind='input')['default_samplerate'])

        self._capture_sr = native_sr
        capture_chunk = int(native_sr * _CHUNK_MS / 1000)

        try:
            self._stream = sd.InputStream(
                device=device_idx,
                channels=1,
                samplerate=native_sr,
                dtype='float32',
                blocksize=capture_chunk,
                callback=self._audio_cb,
                # latency 버퍼를 크게 잡아서 overflow 방지
                latency='high',
            )
            self._stream.start()
            self.get_logger().info(
                f'[Audio] 캡처 시작 — 장치 [{device_idx}] ({self._device_name}), '
                f'{native_sr}Hz → {_TARGET_SR}Hz, chunk {_CHUNK_MS}ms'
            )
        except Exception as e:
            self.get_logger().error(f'[Audio] 스트림 오픈 실패: {e}')

    # ── sounddevice callback (최대한 빠르게) ──────────────────────────────────

    def _audio_cb(self, indata: np.ndarray, frames: int, time_info, status):
        raw = indata[:, 0].copy()
        try:
            # status는 처리 스레드에서 로깅하도록 raw 데이터와 함께 전달
            self._raw_queue.put_nowait((raw, str(status) if status else None))
        except queue.Full:
            pass  # 처리 스레드가 밀릴 경우 조용히 드롭

    # ── 처리 스레드 (리샘플링 + VAD + 발행) ──────────────────────────────────

    def _process_loop(self):
        silence_limit_chunks = int(
            self._speech_timeout * 1000 / _CHUNK_MS
        )
        while rclpy.ok():
            try:
                raw, sd_status = self._raw_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if sd_status:
                self.get_logger().warn(f'[Audio] {sd_status}')

            # 리샘플링
            chunk = _resample(raw, self._capture_sr, _TARGET_SR)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            is_voice = rms > self._vad_thresh

            # VAD 토픽 발행
            vad_msg = Bool()
            vad_msg.data = is_voice
            self.pub_vad.publish(vad_msg)

            if is_voice:
                self._buffer.append(chunk)
                self._silence_chunks = 0
                if not self._speaking:
                    self._speaking = True
                    self.get_logger().debug(f'[Audio] 발화 시작 (RMS {rms:.4f})')
            elif self._speaking:
                self._buffer.append(chunk)
                self._silence_chunks += 1
                if self._silence_chunks >= silence_limit_chunks:
                    self._flush_buffer()

            # 최대 길이 초과 → 강제 플러시
            total = sum(len(c) for c in self._buffer)
            if total >= int(_TARGET_SR * self._max_dur):
                self._flush_buffer()

    def _flush_buffer(self):
        if not self._buffer:
            return

        audio = np.concatenate(self._buffer, axis=0)
        self._buffer.clear()
        self._speaking = False
        self._silence_chunks = 0

        dur = len(audio) / _TARGET_SR
        if dur < self._min_dur:
            self.get_logger().debug(f'[Audio] 발화 {dur:.2f}s — 최소 길이 미달, 무시')
            return

        self.get_logger().info(f'[Audio] 발화 발행 — {dur:.2f}s ({len(audio)} 샘플)')

        msg = Float32MultiArray()
        msg.layout.dim.append(MultiArrayDimension(
            label='samples',
            size=len(audio),
            stride=len(audio),
        ))
        msg.data = audio.tolist()
        self.pub_raw.publish(msg)

    # ── 종료 ──────────────────────────────────────────────────────────────────

    def destroy_node(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HaruAudioNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('[Audio] 종료')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
