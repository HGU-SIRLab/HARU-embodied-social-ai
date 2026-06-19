import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Bool, Float32MultiArray
import json
import time
import threading
from dynamixel_sdk import *

# ── 위치 제어 관절 (Protocol 2.0, Operating Mode 3) ───────────────────────────
HARU_LIMITS = {
    "r_arm_pitch":     {"id": 3,  "min": 1024, "max": 2451},
    "l_arm_pitch":     {"id": 4,  "min": 37,   "max": 1542},
    "r_shoulder_roll": {"id": 5,  "min": 1000, "max": 2050},
    "r_elbow_pitch":   {"id": 6,  "min": 2047, "max": 3062},
    "l_shoulder_roll": {"id": 7,  "min": 1047, "max": 2056},
    "l_elbow_pitch":   {"id": 8,  "min": 1021, "max": 2007},
    "head_pan":        {"id": 10, "min": 1043, "max": 3071},
    "head_tilt":       {"id": 11, "min": 1500, "max": 3086},
    "head_roll":       {"id": 12, "min": 1630, "max": 2452},
}

# ── 속도 제어 바퀴 (Protocol 2.0, Operating Mode 1) ──────────────────────────
WHEEL_LIMITS = {
    "right_wheel": {"id": 1, "min": -300, "max": 300},
    "left_wheel":  {"id": 2, "min": -300, "max": 300},
}

ADDR_GOAL_POSITION   = 116
ADDR_PRESENT_POSITION = 132
ADDR_GOAL_VELOCITY   = 104
ADDR_OPERATING_MODE  = 11
ADDR_TORQUE_ENABLE   = 64
ADDR_PROFILE_ACC     = 108
ADDR_PROFILE_VEL     = 112


class HaruActionNode(Node):
    def __init__(self):
        super().__init__('haru_action_node')

        # hitl_mode=False: brain → haru_vla_raw → action_node (직결)
        # hitl_mode=True : brain → haru_vla_raw → hitl_node → haru_system1_command → action_node
        self.declare_parameter('hitl_mode', False)
        hitl = self.get_parameter('hitl_mode').get_parameter_value().bool_value

        # ── Subscribers ──────────────────────────────────────────────────────
        if hitl:
            self.create_subscription(String, 'haru_system1_command', self._command_callback, 10)
            self.get_logger().info('[Action] HITL 모드: haru_system1_command 구독')
        else:
            self.create_subscription(String, 'haru_vla_raw', self._command_callback, 10)
            self.get_logger().info('[Action] 직결 모드: haru_vla_raw 구독')
        # 수동 테스트용은 항상 구독
        self.create_subscription(String, 'haru_command', self._command_callback, 10)

        # ── Publishers ───────────────────────────────────────────────────────
        self.pub_expression = self.create_publisher(Int32, 'haru_expression', 10)
        # 관절 현재 위치 발행 (10Hz) — hitl_node 키네스테틱 티칭용
        self.pub_joints = self.create_publisher(Float32MultiArray, 'haru_joints/state', 10)

        # ── Subscribers ──────────────────────────────────────────────────────
        # 토크 ON/OFF 제어 (True=ON, False=OFF 키네스테틱 모드)
        self.create_subscription(Bool, 'haru_joints/torque', self._torque_cb, 10)

        # ── Motion state ─────────────────────────────────────────────────────
        self.current_joint_positions = {}
        self.current_wheel_velocities = {k: 0.0 for k in WHEEL_LIMITS}
        self.motion_queue  = []
        self.active_motion = None
        self._kinesthetic  = False   # True 동안 모터 쓰기 중단, 인코더 읽기만
        self._joints_lock  = threading.Lock()

        # ── 50Hz control loop + 10Hz joint state publisher ────────────────────
        self.create_timer(0.02, self._control_loop)
        self.create_timer(0.1,  self._publish_joints)

        # ── Dynamixel 초기화 ──────────────────────────────────────────────────
        self.portHandler   = PortHandler('/dev/ttyACM0')
        self.packetHandler = PacketHandler(2.0)

        if self.portHandler.openPort() and self.portHandler.setBaudRate(57600):
            self.get_logger().info('U2D2 포트 연결 성공')
            self._init_motors()
        else:
            self.get_logger().error('U2D2 포트 연결 실패')

    # ── 초기화 ────────────────────────────────────────────────────────────────

    def _init_motors(self):
        # 위치 제어 관절
        for joint_name, limit in HARU_LIMITS.items():
            dxl_id = limit['id']
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_OPERATING_MODE, 3)  # position
            self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, ADDR_PROFILE_ACC, 0)
            self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, ADDR_PROFILE_VEL, 0)
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)

            pos, result, _ = self.packetHandler.read4ByteTxRx(
                self.portHandler, dxl_id, ADDR_PRESENT_POSITION)
            self.current_joint_positions[joint_name] = (
                pos if result == COMM_SUCCESS
                else (limit['min'] + limit['max']) // 2
            )

        # 바퀴 (속도 제어 모드)
        for wheel_name, limit in WHEEL_LIMITS.items():
            dxl_id = limit['id']
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_OPERATING_MODE, 1)  # velocity
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)
            # 정지 상태로 시작
            self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, ADDR_GOAL_VELOCITY, 0)

        self.get_logger().info('모터 초기화 완료 (관절 9개 + 바퀴 2개)')

    # ── 토크 제어 (키네스테틱 티칭) ──────────────────────────────────────────────

    def _torque_cb(self, msg: Bool):
        enable = bool(msg.data)
        self._kinesthetic = not enable
        val = 1 if enable else 0
        for limit in HARU_LIMITS.values():
            self.packetHandler.write1ByteTxRx(
                self.portHandler, limit['id'], ADDR_TORQUE_ENABLE, val)
        if enable:
            # 토크 재활성 직후 실제 인코더 값으로 current_joint_positions 동기화
            with self._joints_lock:
                for joint_name, limit in HARU_LIMITS.items():
                    pos, result, _ = self.packetHandler.read4ByteTxRx(
                        self.portHandler, limit['id'], ADDR_PRESENT_POSITION)
                    if result == COMM_SUCCESS:
                        self.current_joint_positions[joint_name] = pos
        self.get_logger().info(
            f'[Action] 토크 {"ON (키네스테틱 종료)" if enable else "OFF (키네스테틱 모드 — 자유 조작 가능)"}'
        )

    # ── 관절 상태 발행 (10Hz) ─────────────────────────────────────────────────

    def _publish_joints(self):
        # 키네스테틱 모드에서는 실제 인코더 값을 읽어 발행
        if self._kinesthetic:
            with self._joints_lock:
                for joint_name, limit in HARU_LIMITS.items():
                    pos, result, _ = self.packetHandler.read4ByteTxRx(
                        self.portHandler, limit['id'], ADDR_PRESENT_POSITION)
                    if result == COMM_SUCCESS:
                        self.current_joint_positions[joint_name] = pos

        msg = Float32MultiArray()
        msg.data = [
            float(self.current_joint_positions.get(name, (lim['min'] + lim['max']) // 2))
            for name, lim in HARU_LIMITS.items()
        ]
        self.pub_joints.publish(msg)

    # ── Command callback ──────────────────────────────────────────────────────

    def _command_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.get_logger().info(f"명령 수신: {data.get('speech', '')[:40]!r}")

            # 새 명령 → 기존 큐 클리어 (interrupt)
            self.motion_queue.clear()
            self.active_motion = None

            # expression_id 처리 (display/LED node로 포워딩)
            if 'expression_id' in data:
                expr_msg = Int32()
                expr_msg.data = int(data['expression_id'])
                self.pub_expression.publish(expr_msg)

            # 큐 구성
            if 'sequence' in data and isinstance(data['sequence'], list):
                self.motion_queue = data['sequence']
            elif 'action' in data:
                duration = data.get('duration', 2.0)
                self.motion_queue = [{'action': data['action'], 'duration': duration}]
            else:
                self.motion_queue = [{'action': data, 'duration': 0.5}]

        except Exception as e:
            self.get_logger().error(f'명령 파싱 오류: {e}')

    # ── 50Hz control loop ─────────────────────────────────────────────────────

    def _control_loop(self):
        if self._kinesthetic:
            return  # 키네스테틱 모드: 모터 쓰기 중단 (사용자가 손으로 움직이는 중)
        now = time.time()

        # 1. 큐에서 다음 모션 꺼내기
        if self.active_motion is None:
            if not self.motion_queue:
                # 할 일 없음 → 바퀴 정지
                self._stop_wheels()
                return

            step        = self.motion_queue.pop(0)
            target_act  = step.get('action', {})
            duration    = float(step.get('duration', 0.5))

            # 바퀴 속도 목표값 추출 (나머지는 위치 제어)
            target_wheels = {}
            target_joints = {}
            for k, v in target_act.items():
                if k in WHEEL_LIMITS:
                    target_wheels[k] = float(v)
                elif k in HARU_LIMITS:
                    target_joints[k] = v

            self.active_motion = {
                'start_time':      now,
                'duration':        duration,
                'start_positions': self.current_joint_positions.copy(),
                'target_joints':   target_joints,
                'target_wheels':   target_wheels,
            }

        # 2. 보간 진행률 (0.0 → 1.0)
        elapsed  = now - self.active_motion['start_time']
        duration = self.active_motion['duration']
        progress = min(elapsed / duration, 1.0) if duration > 0 else 1.0

        # 3. Smoothstep 적용 (자연스러운 가감속)
        ease = progress * progress * (3.0 - 2.0 * progress)

        # 4. 위치 제어 관절 보간 → Dynamixel 전송
        for joint_name, target_val in self.active_motion['target_joints'].items():
            limit     = HARU_LIMITS[joint_name]
            start_val = self.active_motion['start_positions'].get(joint_name, target_val)
            try:
                safe_target = int(max(limit['min'], min(int(target_val), limit['max'])))
            except (ValueError, TypeError):
                continue
            current_val = int(start_val + (safe_target - start_val) * ease)
            self.current_joint_positions[joint_name] = current_val
            try:
                self.packetHandler.write4ByteTxRx(
                    self.portHandler, limit['id'], ADDR_GOAL_POSITION, current_val)
            except Exception:
                pass

        # 5. 바퀴 속도 제어 (모션이 활성인 동안 유지)
        for wheel_name, target_vel in self.active_motion['target_wheels'].items():
            limit    = WHEEL_LIMITS[wheel_name]
            safe_vel = int(max(limit['min'], min(int(target_vel), limit['max'])))
            self.current_wheel_velocities[wheel_name] = safe_vel
            try:
                self.packetHandler.write4ByteTxRx(
                    self.portHandler, limit['id'], ADDR_GOAL_VELOCITY, safe_vel)
            except Exception:
                pass

        # 6. 모션 완료 처리
        if progress >= 1.0:
            self.active_motion = None
            self._stop_wheels()  # 모션 끝 → 바퀴 정지

    def _stop_wheels(self):
        for wheel_name, limit in WHEEL_LIMITS.items():
            if self.current_wheel_velocities.get(wheel_name, 0) != 0:
                self.current_wheel_velocities[wheel_name] = 0
                try:
                    self.packetHandler.write4ByteTxRx(
                        self.portHandler, limit['id'], ADDR_GOAL_VELOCITY, 0)
                except Exception:
                    pass


def main(args=None):
    rclpy.init(args=args)
    node = HaruActionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('종료 요청')
    finally:
        # 안전 종료: 전체 모터 토크 OFF
        for limit in HARU_LIMITS.values():
            node.packetHandler.write1ByteTxRx(node.portHandler, limit['id'], ADDR_TORQUE_ENABLE, 0)
        for limit in WHEEL_LIMITS.values():
            node.packetHandler.write4ByteTxRx(node.portHandler, limit['id'], ADDR_GOAL_VELOCITY, 0)
            node.packetHandler.write1ByteTxRx(node.portHandler, limit['id'], ADDR_TORQUE_ENABLE, 0)
        node.portHandler.closePort()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
