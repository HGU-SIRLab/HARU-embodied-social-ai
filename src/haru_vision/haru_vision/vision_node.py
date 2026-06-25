import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

# SSH 환경 등 디스플레이가 없으면 imshow 시도 안 함
# OpenCV가 GTK/Cocoa 없이 빌드된 경우도 헤드리스로 동작
_HAS_DISPLAY = bool(os.environ.get('DISPLAY'))
_DISPLAY_WORKING = _HAS_DISPLAY  # 첫 imshow 실패 시 False로 변경

IMG_SIZE   = 448   # 각 카메라 정방형 크기
JPEG_QUAL  = 80
CAPTURE_HZ = 3.0


class HaruVisionNode(Node):
    def __init__(self):
        super().__init__('haru_vision_node')

        self.publisher_ = self.create_publisher(
            CompressedImage, 'haru_vision/compressed', 10)

        self.cap_head = cv2.VideoCapture(0)  # RealSense (얼굴/시선)
        self.cap_body = cv2.VideoCapture(4)  # C270 (몸통)

        self._head_ok = self.cap_head.isOpened()
        self._body_ok = self.cap_body.isOpened()

        if not self._head_ok:
            self.get_logger().warn('RealSense(0) 없음 — body 카메라만 사용')
        if not self._body_ok:
            self.get_logger().warn('C270(4) 없음 — head 카메라만 사용')
        if not self._head_ok and not self._body_ok:
            self.get_logger().error('카메라 없음!')

        self.create_timer(1.0 / CAPTURE_HZ, self._capture)
        self.get_logger().info(
            f'Vision Node 시작 ({CAPTURE_HZ}Hz, display={"on" if _HAS_DISPLAY else "off"})'
        )

    def _capture(self):
        head_frame = self._read(self.cap_head) if self._head_ok else None
        body_frame = self._read(self.cap_body) if self._body_ok else None

        if head_frame is None and body_frame is None:
            self.get_logger().warn('프레임 읽기 실패')
            return

        # 두 카메라: 좌(얼굴) + 우(몸통) → 896x448
        if head_frame is not None and body_frame is not None:
            merged = cv2.hconcat([head_frame, body_frame])
        elif head_frame is not None:
            # head만 있으면 양쪽에 복제 (brain에 일정한 해상도 보장)
            merged = cv2.hconcat([head_frame, head_frame])
        else:
            merged = cv2.hconcat([body_frame, body_frame])

        global _DISPLAY_WORKING
        if _DISPLAY_WORKING:
            try:
                cv2.imshow('HARU Vision', merged)
                cv2.waitKey(1)
            except cv2.error:
                _DISPLAY_WORKING = False
                self.get_logger().warn('cv2.imshow 불가 — 헤드리스 모드로 전환')

        ok, buf = cv2.imencode('.jpg', merged,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUAL])
        if not ok:
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        self.publisher_.publish(msg)

    @staticmethod
    def _read(cap: cv2.VideoCapture):
        ret, frame = cap.read()
        if not ret:
            return None
        return cv2.resize(frame, (IMG_SIZE, IMG_SIZE))


def main(args=None):
    rclpy.init(args=args)
    node = HaruVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap_head.release()
        node.cap_body.release()
        if _DISPLAY_WORKING:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
