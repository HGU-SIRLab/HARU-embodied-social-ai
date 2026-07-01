"""
HARU 표정 디스플레이 노드
/haru_expression (Int32) → Robot-display-HRI pygame 얼굴 애니메이션
표정 ID: 0=neutral 1=joy 2=sadness 3=curiosity 4=surprise 5=empathy 6=thinking 7=concern
"""
import os
import queue
import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

EMOTION_MAP = {
    0: "NEUTRAL",    # neutral
    1: "HAPPY",      # joy
    2: "SAD",        # sadness
    3: "THINKING",   # curiosity
    4: "SURPRISED",  # surprise
    5: "TENDER",     # empathy
    6: "THINKING",   # thinking
    7: "SCARED",     # concern
}


class HaruExpressionNode(Node):
    def __init__(self, emotion_queue: queue.Queue):
        super().__init__('haru_expression_node')
        self._q = emotion_queue
        self.create_subscription(Int32, 'haru_expression', self._on_expression, 10)
        self.get_logger().info('Expression Node 시작 — Robot-display-HRI pygame')

    def _on_expression(self, msg: Int32):
        key = EMOTION_MAP.get(msg.data, "NEUTRAL")
        self.get_logger().info(f'[Expr] {msg.data} → {key}')
        self._q.put(key)


def main(args=None):
    os.environ.setdefault('DISPLAY', ':0')

    # robot_face 패키지를 import path에 추가 (symlink install 대응: realpath 사용)
    robot_face_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'robot_face')
    if robot_face_dir not in sys.path:
        sys.path.insert(0, robot_face_dir)

    rclpy.init(args=args)

    emotion_q = queue.Queue()
    stop_event = threading.Event()
    node = HaruExpressionNode(emotion_q)

    spin_thread = threading.Thread(
        target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()

    try:
        from main import run_face_app
        run_face_app(emotion_q, stop_event, None)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == '__main__':
    main()
