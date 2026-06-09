import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

class HaruVisionNode(Node):
    def __init__(self):
        super().__init__('haru_vision_node')
        
        # 뇌(brain_node)로 전송할 이미지 퍼블리셔
        self.publisher_ = self.create_publisher(CompressedImage, 'haru_vision/compressed', 10)
        
        # 듀얼 카메라 포트 설정 (v4l2-ctl 결과 반영)
        self.cap_head = cv2.VideoCapture(0) # RealSense (얼굴/시선)
        self.cap_body = cv2.VideoCapture(4) # C270 (몸통/동작)
        
        if not self.cap_head.isOpened():
            self.get_logger().error('❌ RealSense(포트 0)를 열 수 없습니다!')
        if not self.cap_body.isOpened():
            self.get_logger().error('❌ C270(포트 4)를 열 수 없습니다!')

        # 1초에 3프레임(3Hz) 간격으로 사진 캡처
        self.timer = self.create_timer(1.0 / 3.0, self.capture_and_publish)
        self.get_logger().info('👁️👁️ HARU 듀얼 시각 신경망 가동! (RealSense + C270)')

    def capture_and_publish(self):
        ret_head, frame_head = self.cap_head.read()
        ret_body, frame_body = self.cap_body.read()
        
        if ret_head and ret_body:
            # 1. 크기 맞추기 (VLM 최적화를 위해 둘 다 448x448로 조절)
            head_resized = cv2.resize(frame_head, (448, 448))
            body_resized = cv2.resize(frame_body, (448, 448))
            
            # 2. 두 영상을 좌우로 이어 붙이기 (왼쪽: 얼굴, 오른쪽: 몸통) -> 결과 해상도 896x448
            merged_frame = cv2.hconcat([head_resized, body_resized])
            
            # 3. 모니터에 로봇의 시야 띄우기 (디버깅용 창)
            try:
                cv2.imshow('HARU Dual Vision (Left: Head, Right: Body)', merged_frame)
                cv2.waitKey(1) # OpenCV 창 업데이트를 위한 필수 대기 시간 (1ms)
            except Exception as e:
                # SSH 환경 등 모니터 출력이 불가능할 경우 노드가 죽지 않도록 방어
                pass

            # 4. ROS2 메시지 포맷으로 변환 (JPEG 압축)
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            _, compressed_data = cv2.imencode('.jpg', merged_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            msg.data = compressed_data.tobytes()
            
            # 5. 토픽 발행 (대뇌로 쏘기)
            self.publisher_.publish(msg)
        else:
            self.get_logger().warning('⚠️ 카메라에서 프레임을 읽어오지 못했습니다.')

def main(args=None):
    rclpy.init(args=args)
    node = HaruVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 카메라 및 창 깔끔하게 닫기
        node.cap_head.release()
        node.cap_body.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()