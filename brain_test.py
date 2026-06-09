import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import json

# ==========================================
# 1. Qwen3-VL-8B 전용 로딩 (안전 모드 장착)
# ==========================================
model_id = "Qwen/Qwen3-VL-8B-Instruct" 
print(f"[{model_id}] HRI 두뇌를 깨우는 중입니다...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="eager",  # <-- [핵심] PyTorch SDPA 버전 충돌(enable_gqa) 원천 차단
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
print("두뇌 로딩 완료!\n")

# ==========================================
# 2. HRI 입력 설정
# ==========================================
system_prompt = """
당신은 사람과 감정을 교류하는 친근하고 따뜻한 로봇입니다.
카메라(비전)로 들어온 상황과 사용자의 말을 분석하여 응답하십시오.
반드시 아래의 JSON 포맷으로만 응답해야 하며, 다른 부연 설명은 절대 하지 마십시오.
{
  "speech": "대사",
  "action": {"motion_type": "greet", "head_pan": 0, "head_tilt": 0}
}
"""
test_image = Image.new('RGB', (224, 224), color='black')
user_text = "안녕? 내가 보이니?"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": [
        {"type": "image", "image": test_image},
        {"type": "text", "text": user_text}
    ]}
]

# ==========================================
# 3. 추론 실행
# ==========================================
print("응답 생성 중...")
text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text_input],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to("cuda")

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

print("\n=== AI 출력 ===\n", output_text)