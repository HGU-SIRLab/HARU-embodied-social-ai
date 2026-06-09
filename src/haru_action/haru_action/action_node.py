import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
from dynamixel_sdk import *

# --- HARU 모터 스펙 및 한계치 ---
HARU_LIMITS = {
    "r_arm_pitch": {"id": 3, "min": 1024, "max": 2451},
    "l_arm_pitch": {"id": 4, "min": 37, "max": 1542},
    "r_shoulder_roll": {"id": 5, "min": 1000, "max": 2050},
    "r_elbow_pitch": {"id": 6, "min": 2047, "max": 3062},
    "l_shoulder_roll": {"id": 7, "min": 1047, "max": 2056},
    "l_elbow_pitch": {"id": 8, "min": 1021, "max": 2007},
    "head_pan": {"id": 10, "min": 1043, "max": 3071},
    "head_tilt": {"id": 11, "min": 1500, "max": 3086},
    "head_roll": {"id": 12, "min": 1630, "max": 2452}
}

class HaruActionNode(Node):
    def __init__(self):
        super().__init__('haru_action_node')
        
        self.subscription = self.create_subscription(String, 'haru_command', self.command_callback, 10)
        
        # --- [추가] 궤적 제어 및 상태 저장용 변수 ---
        self.current_joint_positions = {} # 로봇이 현재 있다고 믿는 각도
        self.motion_queue = []            # 앞으로 해야 할 시퀀스 대기열
        self.active_motion = None         # 지금 실행 중인 모션
        
        # --- [추가] 50Hz (0.02초) 주기로 궤적을 쏘는 척수의 심장박동! ---
        self.timer = self.create_timer(0.02, self.control_loop)

        # 다이나믹셀 세팅
        self.portHandler = PortHandler('/dev/ttyACM0')
        self.packetHandler = PacketHandler(2.0)
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132  # 현재 위치 읽기 주소
        
        if self.portHandler.openPort() and self.portHandler.setBaudRate(57600):
            self.get_logger().info('✅ U2D2 포트 연결 성공!')
            self.init_motors()
        else:
            self.get_logger().error('❌ U2D2 포트 연결 실패!')

    def init_motors(self):
        # [수정] 소프트웨어가 직접 스플라인 보간을 하므로, 하드웨어 딜레이는 0(최대 속도/가속도)으로 풉니다!
        ADDR_PROFILE_ACCELERATION = 108
        ADDR_PROFILE_VELOCITY = 112
        PROFILE_ACC_VALUE = 0   
        PROFILE_VEL_VALUE = 0  

        for joint_name, limit in HARU_LIMITS.items():
            dxl_id = limit['id']
            
            # 1. EEPROM 락 방지용 토크 OFF
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, 64, 0) 
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, 11, 3) # 위치 제어 모드
            
            # 2. 하드웨어 스무딩 해제 (소프트웨어 궤적이 컨트롤함)
            self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, ADDR_PROFILE_ACCELERATION, PROFILE_ACC_VALUE)
            self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, ADDR_PROFILE_VELOCITY, PROFILE_VEL_VALUE)
            
            # 3. 토크 ON
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, 64, 1) 
            
            # 4. [추가] 켜질 때 튀지 않도록 모터의 '현재 위치'를 읽어서 상태 변수에 저장
            pos, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, dxl_id, self.ADDR_PRESENT_POSITION)
            if dxl_comm_result == COMM_SUCCESS:
                self.current_joint_positions[joint_name] = pos
            else:
                self.current_joint_positions[joint_name] = (limit['min'] + limit['max']) // 2
                
        self.get_logger().info('🩰 비동기 스플라인 궤적 제어 엔진 초기화 완료!')

    def command_callback(self, msg):
        try:
            command_data = json.loads(msg.data)
            self.get_logger().info(f"📥 새 명령 수신: {command_data.get('speech', '무음')}")

            # 💡 [핵심] 인터럽트 (Interrupt) 발생!
            # 새로운 명령이 오면 예전에 하려던 동작들을 싹 지워버립니다.
            self.motion_queue.clear()
            self.active_motion = None 

            if "sequence" in command_data and isinstance(command_data["sequence"], list):
                self.motion_queue = command_data["sequence"]
            elif "action" in command_data:
                self.motion_queue = [{"action": command_data["action"], "duration": 0.5}]
            else:
                self.motion_queue = [{"action": command_data, "duration": 0.5}]

        except Exception as e:
            self.get_logger().error(f"명령 파싱 오류: {e}")

    def control_loop(self):
        """0.02초마다 호출되며 선형 보간(Spline) 궤적을 모터로 쏘는 함수"""
        now = time.time()

        # 1. 현재 실행 중인 동작이 없다면 큐에서 새로 꺼내기
        if self.active_motion is None:
            if not self.motion_queue:
                return # 할 일이 없으면 휴식
            
            next_step = self.motion_queue.pop(0)
            target_action = next_step.get("action", {})
            duration = next_step.get("duration", 0.5)
            
            self.active_motion = {
                "start_time": now,
                "duration": duration,
                "start_positions": self.current_joint_positions.copy(), # 현재 위치를 출발점으로!
                "target_positions": target_action
            }

        # 2. 보간(Interpolation) 진행률 계산 (0.0 ~ 1.0)
        elapsed = now - self.active_motion["start_time"]
        progress = elapsed / self.active_motion["duration"] if self.active_motion["duration"] > 0 else 1.0
        
        if progress >= 1.0:
            progress = 1.0

        # 3. 💡 스무스스텝 (Smoothstep) 함수 적용 (자연스러운 가감속)
        # 3t^2 - 2t^3 공식을 사용해 시작과 끝을 부드럽게 깎아줍니다.
        ease = progress * progress * (3.0 - 2.0 * progress)

        # 4. 각 관절별로 부드럽게 쪼개진 각도 계산 및 전송
        for joint_name, target_val in self.active_motion["target_positions"].items():
            if joint_name in HARU_LIMITS:
                limit = HARU_LIMITS[joint_name]
                start_val = self.active_motion["start_positions"].get(joint_name, target_val)
                
                # 안전 클리핑 (목표값 자체를 제한)
                try:
                    safe_target = int(max(limit["min"], min(int(target_val), limit["max"])))
                except ValueError:
                    continue
                
                # 시작점과 끝점 사이의 현재 시점 각도 계산
                current_val = int(start_val + (safe_target - start_val) * ease)
                
                # 상태 업데이트 및 전송
                self.current_joint_positions[joint_name] = current_val
                self.packetHandler.write4ByteTxRx(self.portHandler, limit['id'], self.ADDR_GOAL_POSITION, current_val)

        # 5. 모션이 끝났으면 비워주기 (다음 루프에서 큐의 다음 동작을 꺼냄)
        if progress >= 1.0:
            self.active_motion = None


def main(args=None):
    rclpy.init(args=args)
    node = HaruActionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('프로그램 종료 요청됨.')
    finally:
        for joint in HARU_LIMITS.values():
            node.packetHandler.write1ByteTxRx(node.portHandler, joint['id'], 64, 0)
        node.portHandler.closePort()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()