import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
import json
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import cv2
import numpy as np
from collections import deque

class HaruBrainNode(Node):
    def __init__(self):
        super().__init__('haru_brain_node')
        
        # 1. 척수로 명령을 내릴 퍼블리셔
        self.publisher_ = self.create_publisher(String, 'haru_command', 10)
        
        # 2. 시각 신경망에서 사진을 받아올 서브스크라이버
        self.subscription = self.create_subscription(
            CompressedImage,
            'haru_vision/compressed',
            self.vision_callback,
            10
        )
        
        # 3. 비디오 인식을 위한 프레임 버퍼 (최근 3장 유지)
        self.frame_buffer = deque(maxlen=3)
        self.is_thinking = False # 뇌 과부하 방지 락(Lock)

        self.get_logger().info('🧠 HARU 대뇌 노드 가동 중... (VLM 로딩 중)')
        
        # Qwen3-VL 모델 로딩
        model_id = "Qwen/Qwen3-VL-8B-Instruct" 
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cuda",
            attn_implementation="eager", trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.get_logger().info('✅ 대뇌 로딩 완료! 카메라 데이터를 기다립니다.')
        
        # 4. 0.5초마다 뇌를 깨워서 번개처럼 반응하게 만드는 루프
        self.timer = self.create_timer(0.5, self.think_and_act)

    def vision_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            self.frame_buffer.append(pil_image)
        except Exception as e:
            self.get_logger().error(f"이미지 디코딩 실패: {e}")

    def think_and_act(self):
        if self.is_thinking or len(self.frame_buffer) < 3:
            return

        self.is_thinking = True
        self.get_logger().info('🤔 동작 인지 및 자율 궤적 설계 중...')

        frames = list(self.frame_buffer)

        # 자율 궤적 생성 및 듀얼 비전(Dual Vision) 융합 시스템 프롬프트
        system_prompt = """
        당신은 사람과 교감하는 피지컬 AI 로봇 'HARU(하루)'입니다.
        당신에게는 1초 동안 촬영된 3장의 사진(비디오 프레임)이 주어집니다.
        
        [👁️ 시각 데이터(이미지) 구조 안내]
        주어진 이미지는 2대의 카메라 화면을 좌우로 결합한(Concatenated) 형태입니다.
        - 왼쪽 절반 (RealSense 카메라): 사용자의 얼굴, 표정, 시선을 보여줍니다.
        - 오른쪽 절반 (C270 카메라): 사용자의 몸통, 손짓, 전체적인 제스처를 보여줍니다.

        [동작 설계 지침]
        당신은 이 '두 가지 정보'를 융합하여 상황을 분석해야 합니다.
        1. 시선과 교감: 사용자가 나(로봇)를 똑바로 쳐다보며(왼쪽 화면) 손을 흔든다면(오른쪽 화면), 매우 반갑게 양팔을 모두 사용하여 시퀀스를 짜십시오.
        2. 자율 판단: 사용자가 화면의 어느 쪽(좌/우)에 치우쳐 있는지 파악하여, 가까운 쪽 팔을 우선적으로 사용해 인사하십시오.
        3. 역동성: 어깨(shoulder_roll)를 여러 번 왕복시키는 연속 동작(Sequence)을 만들어 생동감을 주십시오.
        4. 가만히 있기: 사용자가 손을 흔들지 않고 가만히 쳐다만 보거나 무의미한 행동을 한다면 빈 sequence([])를 반환하십시오.

        [당신의 신체 구조]
        - 오른팔: r_arm_pitch (1024~2451), r_shoulder_roll (1000~2050)
        - 왼팔: l_arm_pitch (37~1542), l_shoulder_roll (1047~2056)
        - 고개: head_pan (1043~3071), head_tilt (1500~3086)

        반드시 아래의 JSON 포맷으로만 응답하십시오.
        {
          "speech": "짧은 대사!",
          "sequence": [
            {"action": {"r_arm_pitch": 2400, "r_shoulder_roll": 1200}, "duration": 1.0},
            {"action": {"r_shoulder_roll": 1800}, "duration": 0.5},
            {"action": {"r_shoulder_roll": 1200}, "duration": 0.5},
            {"action": {"r_arm_pitch": 1024, "r_shoulder_roll": 2050}, "duration": 1.0}
          ]
        }
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": frames[0]},
                {"type": "image", "image": frames[1]},
                {"type": "image", "image": frames[2]},
                {"type": "text", "text": "사용자의 행동을 분석하고, 나(HARU)의 신체를 활용한 연속 동작 JSON을 생성해 줘!"}
            ]}
        ]
        
        try:
            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text_input], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                # 시퀀스 데이터가 길어질 수 있으므로 max_new_tokens를 150으로 설정 (JSON 잘림 방지)
                generated_ids = self.model.generate(**inputs, max_new_tokens=150)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            
            self.get_logger().info(f'💡 VLM 생성 궤적:\n{output_text}')
            
            # JSON 텍스트 정제 후 전송
            json_str = output_text.replace("```json", "").replace("```", "").strip()
            msg = String()
            msg.data = json_str
            self.publisher_.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'추론 에러 발생: {e}')
        finally:
            self.frame_buffer.clear()
            self.is_thinking = False

def main(args=None):
    rclpy.init(args=args)
    node = HaruBrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()