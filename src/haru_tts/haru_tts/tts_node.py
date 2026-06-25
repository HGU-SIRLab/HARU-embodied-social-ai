import asyncio
import os
import subprocess
import tempfile
import threading
import time

import edge_tts
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

TTS_VOICE   = 'ko-KR-SunHiNeural'   # 한국어 여성 자연음성
AUDIO_DEVICE = os.environ.get('HARU_AUDIO_OUT', 'default')  # hw:3,0 or default
TTS_SPEED    = '-10%'    # 약간 느리게 (로봇 특성상 또렷하게)

class HaruTTSNode(Node):
    def __init__(self):
        super().__init__('haru_tts_node')

        self._sub = self.create_subscription(
            String, 'haru_speech', self._on_speech, 10)

        # 현재 재생 중인 mpg123 프로세스
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

        # asyncio 루프를 별도 스레드에서 실행 (ROS2 spin과 분리)
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        self.get_logger().info(
            f'TTS Node 시작 — 음성: {TTS_VOICE}, 출력: {AUDIO_DEVICE}')

    def _on_speech(self, msg: String):
        text = msg.data.strip()
        if not text:
            return  # speech="" 침묵 선택 무시
        asyncio.run_coroutine_threadsafe(self._speak(text), self._loop)

    async def _speak(self, text: str):
        # 기존 재생 중단
        self._stop_playback()

        tmp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            t0 = time.monotonic()
            tts = edge_tts.Communicate(text, voice=TTS_VOICE, rate=TTS_SPEED)
            await tts.save(tmp_path)
            gen_ms = (time.monotonic() - t0) * 1000
            self.get_logger().info(
                f'[TTS] 생성 {gen_ms:.0f}ms | "{text[:30]}{"..." if len(text)>30 else ""}"')

            self._play(tmp_path)
        except Exception as e:
            self.get_logger().error(f'[TTS] 오류: {e}')
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _play(self, mp3_path: str):
        cmd = ['mpg123', '-q', '-a', AUDIO_DEVICE, mp3_path]
        with self._lock:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        # 재생 완료 후 파일 삭제 (별도 스레드)
        threading.Thread(
            target=self._wait_and_cleanup,
            args=(self._proc, mp3_path),
            daemon=True
        ).start()

    def _wait_and_cleanup(self, proc: subprocess.Popen, path: str):
        proc.wait()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    def _stop_playback(self):
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                self._proc = None

    def destroy_node(self):
        self._stop_playback()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HaruTTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
