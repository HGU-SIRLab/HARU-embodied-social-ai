# HARU: A Physical AI Robot with Vision-Language Model for Human-Robot Interaction

**Team:** Cho Hyeongmin

**YouTube Demo:** [▶ Watch Demo Video](https://youtu.be/XXXXXXXXXXXX)

---

## 1. Introduction

Recent advances in Vision-Language Models (VLMs) and large-scale robot learning have opened new possibilities for building robots that can naturally interact with humans without hand-crafted rules. Works such as OpenVLA, π0.5, HumanPlus, and Diffusion Policy have demonstrated that end-to-end learned policies can generalize across diverse tasks and environments.

Inspired by these approaches, this project presents **HARU** (Human-Aware Responsive Unit), a physical AI robot designed for expressive Human-Robot Interaction (HRI). Instead of pre-programmed motion sequences, HARU uses a Vision-Language Model to *observe* the human in real time and *autonomously generate* a natural response — both a verbal reply and a full motor action sequence — on the fly.

The key idea is simple: **give the robot eyes, a brain, and a body, and let the AI decide how to react.**

---

## 2. System Architecture

HARU is built on ROS2 and consists of three tightly coupled nodes:

```
[Dual Camera]
      │
      ▼
 haru_vision ──(haru_vision/compressed)──▶ haru_brain ──(haru_command)──▶ haru_action
      │                                         │                               │
RealSense + C270                         Qwen3-VL-8B                    Dynamixel Motors
(face & body)                          (VLM Inference)                  (9 joints, 50Hz)
```

### 2.1 haru_vision (Eyes)
- Captures from two cameras simultaneously:
  - **Port 0 — Intel RealSense**: user's face and gaze
  - **Port 4 — Logitech C270**: user's torso and hand gestures
- Resizes both frames to **448×448** and concatenates them horizontally → **896×448** single image
- Publishes at **3 Hz** as JPEG-compressed ROS2 `CompressedImage` messages

### 2.2 haru_brain (Brain)
- Subscribes to the vision topic and maintains a **rolling buffer of 3 frames** (≈1 second of video)
- Every **0.5 seconds**, feeds the 3-frame buffer into **Qwen3-VL-8B-Instruct** (float16, CUDA)
- The system prompt instructs the model to:
  - Fuse left-half (face/gaze) and right-half (body/gesture) information
  - Autonomously design a natural motor action sequence (`sequence`)
  - Return a strictly structured JSON with speech and motion

**Example VLM Output:**
```json
{
  "speech": "Hi there! Nice to see you!",
  "sequence": [
    {"action": {"r_arm_pitch": 2400, "r_shoulder_roll": 1200}, "duration": 1.0},
    {"action": {"r_shoulder_roll": 1800}, "duration": 0.5},
    {"action": {"r_shoulder_roll": 1200}, "duration": 0.5},
    {"action": {"r_arm_pitch": 1024, "r_shoulder_roll": 2050}, "duration": 1.0}
  ]
}
```

### 2.3 haru_action (Body)
- Subscribes to `haru_command` and drives **9 Dynamixel servos** via U2D2 adapter (`/dev/ttyACM0`, 57600 baud)
- Runs a **50 Hz control loop** that executes the received sequence step by step
- Applies **Smoothstep interpolation** (`3t² - 2t³`) for natural acceleration and deceleration
- New commands **interrupt** the current motion immediately for responsive interaction

**Controllable Joints:**

| Joint | Dynamixel ID | Range |
|---|---|---|
| r_arm_pitch | 3 | 1024 ~ 2451 |
| l_arm_pitch | 4 | 37 ~ 1542 |
| r_shoulder_roll | 5 | 1000 ~ 2050 |
| r_elbow_pitch | 6 | 2047 ~ 3062 |
| l_shoulder_roll | 7 | 1047 ~ 2056 |
| l_elbow_pitch | 8 | 1021 ~ 2007 |
| head_pan | 10 | 1043 ~ 3071 |
| head_tilt | 11 | 1500 ~ 3086 |
| head_roll | 12 | 1630 ~ 2452 |

---

## 3. Hardware & Software Requirements

### 3.1 하드웨어 구성

HARU를 실행하려면 아래 하드웨어가 모두 필요합니다.

| 구성 요소 | 모델 | 역할 | 연결 방법 |
|---|---|---|---|
| 메인 컴퓨터 | NVIDIA Jetson (aarch64, JetPack) | 전체 시스템 실행 | — |
| 얼굴 카메라 | Intel RealSense D435 | 사용자 얼굴·시선 감지 | USB → `/dev/video0` |
| 몸통 카메라 | Logitech C270 | 사용자 몸통·손 제스처 감지 | USB → `/dev/video4` |
| 모터 어댑터 | ROBOTIS U2D2 | Dynamixel 통신 변환기 (USB↔TTL) | USB → `/dev/ttyACM0` |
| 서보 모터 | Dynamixel XL / XM 계열 × 9 | 관절 구동 | 3핀 TTL 체인 → U2D2 |
| 전원 공급 | 12V DC (Dynamixel 규격) | 모터 전원 | SMPS → 파워 허브 → 각 모터 |

**하드웨어 연결 구성도:**

```
[Jetson USB 포트]
   ├── USB ──▶ Intel RealSense D435 (/dev/video0)   ← 얼굴/시선 카메라
   ├── USB ──▶ Logitech C270        (/dev/video4)   ← 몸통/손짓 카메라
   └── USB ──▶ U2D2 어댑터         (/dev/ttyACM0)  ← Dynamixel 통신

[U2D2 TTL 체인] (57600 baud, Protocol 2.0)
   U2D2 ──▶ ID:3  r_arm_pitch    (우 팔 피치)
         ──▶ ID:4  l_arm_pitch    (좌 팔 피치)
         ──▶ ID:5  r_shoulder_roll (우 어깨 롤)
         ──▶ ID:6  r_elbow_pitch  (우 팔꿈치)
         ──▶ ID:7  l_shoulder_roll (좌 어깨 롤)
         ──▶ ID:8  l_elbow_pitch  (좌 팔꿈치)
         ──▶ ID:10 head_pan       (고개 좌우)
         ──▶ ID:11 head_tilt      (고개 상하)
         ──▶ ID:12 head_roll      (고개 기울기)

[12V 전원]
   SMPS ──▶ 파워 허브 ──▶ 모든 Dynamixel 모터 (데이지 체인)
```

> Dynamixel 모터들은 3핀 케이블로 직렬 데이지 체인(daisy-chain) 연결됩니다.
> U2D2는 Jetson과 모터 체인 사이의 USB↔TTL 변환기 역할을 합니다.

---

### 3.2 소프트웨어 구성

**필수 소프트웨어:**

| 소프트웨어 | 버전 | 설치 방법 |
|---|---|---|
| JetPack SDK | 6.x (aarch64) | NVIDIA 공식 이미지 플래싱 |
| ROS2 | Humble Hawksbill | `apt install ros-humble-desktop` |
| Python | 3.10 | JetPack 기본 포함 |
| CUDA | 12.x | JetPack 기본 포함 |
| PyTorch | 2.5.0 (Jetson 전용) | 아래 설치 가이드 참고 |
| Qwen3-VL-8B-Instruct | — | HuggingFace 자동 다운로드 |

**Python 패키지 (`requirements.txt`):**

```
transformers>=5.8.0
qwen-vl-utils>=0.0.14
accelerate>=0.27.0
safetensors>=0.4.0
huggingface-hub>=0.23.0
opencv-python>=4.8.1
Pillow>=10.0.0
numpy>=1.26.0
```

---

### 3.3 설치 가이드

#### Step 1 — ROS2 Humble 설치

```bash
# ROS2 Humble 설치 (Ubuntu 22.04 / JetPack 기준)
sudo apt update && sudo apt install -y ros-humble-desktop
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

#### Step 2 — PyTorch 설치 (Jetson 전용 wheel)

```bash
# 워크스페이스 루트에 있는 Jetson 전용 wheel 파일로 설치
cd ~/robot_brain_workspace
pip install torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

> 일반 `pip install torch`는 x86용이므로 Jetson에서 CUDA 가속이 동작하지 않습니다.
> 반드시 위의 Jetson 전용 wheel 파일을 사용하십시오.

#### Step 3 — Python 패키지 설치

```bash
pip install -r requirements.txt
```

#### Step 4 — ROS2 워크스페이스 빌드

```bash
cd ~/robot_brain_workspace
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

#### Step 5 — VLM 모델 다운로드 확인

`haru_brain` 노드 최초 실행 시 HuggingFace에서 `Qwen/Qwen3-VL-8B-Instruct`가 자동으로 다운로드됩니다 (약 16GB). 미리 받아두려면:

```bash
python3 -c "from transformers import Qwen3VLForConditionalGeneration; Qwen3VLForConditionalGeneration.from_pretrained('Qwen/Qwen3-VL-8B-Instruct')"
```

---

### 3.4 실행 방법

터미널을 **3개** 열어서 각 노드를 순서대로 실행합니다.

```bash
# 공통 — 모든 터미널에서 먼저 실행
source /opt/ros/humble/setup.bash
source ~/robot_brain_workspace/install/setup.bash
```

```bash
# 터미널 1 — 눈: 카메라 영상 캡처 및 발행
ros2 run haru_vision vision_node
```

```bash
# 터미널 2 — 뇌: VLM 추론 및 명령 생성 (모델 로딩에 약 30~60초 소요)
ros2 run haru_brain brain_node
```

```bash
# 터미널 3 — 몸: 모터 제어 실행
ros2 run haru_action action_node
```

**VLM 단독 테스트 (ROS2 없이):**

```bash
# 모델 로딩과 추론만 단독으로 테스트
cd ~/robot_brain_workspace
python3 brain_test.py
```

---

## 5. Task and Method

### Task
The core task is **real-time gesture-driven HRI**: given a continuous video stream of a person, HARU must recognize the person's intent and respond with a contextually appropriate, expressive motor action — without any pre-defined if-else logic.

### Method

**Dual Vision Fusion**
The two camera feeds are concatenated into a single wide image before being passed to the VLM. This allows the model to simultaneously reason about facial expressions/gaze (left half) and full-body gestures (right half) in a single forward pass, avoiding the need for separate perception modules.

**VLM as Motion Planner**
Rather than using the VLM purely for perception and a separate planner for action (as in OpenVLA), HARU delegates the entire decision — *what to say* and *how to move* — to the VLM. The model outputs a full joint-space trajectory as a JSON sequence, which the action node executes directly.

**Software Trajectory Control**
Dynamixel hardware profile acceleration/velocity are set to 0, giving full trajectory authority to the software. The action node interpolates between keyframes using Smoothstep, producing smooth and natural-looking motion at 50 Hz.

---

## 6. Experiments

### 6.1 Setup
| Component | Spec |
|---|---|
| Platform | NVIDIA Jetson (aarch64, JetPack) |
| VLM | Qwen3-VL-8B-Instruct (float16) |
| Framework | ROS2, PyTorch 2.4/2.5 |
| Cameras | Intel RealSense D435 + Logitech C270 |
| Motors | Dynamixel Protocol 2.0 (×9) |

### 6.2 Evaluated Scenarios
The following interaction scenarios were tested:

| Scenario | Expected HARU Response |
|---|---|
| User waves hand directly at robot | Both arms wave back with greeting speech |
| User waves from the left side | Left arm preferentially raised |
| User waves from the right side | Right arm preferentially raised |
| User stands still / no gesture | Empty sequence returned, no movement |
| User makes eye contact only | Subtle head movement, short verbal response |

### 6.3 Latency Analysis
| Stage | Measured Time |
|---|---|
| Camera capture → ROS publish | ~33 ms (3 Hz) |
| Frame buffer fill (3 frames) | ~1 sec |
| Qwen3-VL-8B inference (Jetson) | ~3–5 sec |
| JSON parse → first motor command | < 5 ms |
| **Total end-to-end latency** | **~4–6 sec** |

---

## 7. Result Analysis

HARU successfully demonstrated **context-aware, autonomous motion generation** driven entirely by a VLM. Key observations:

- **Gesture recognition worked reliably** for clear, intentional gestures (waving, pointing). Subtle or ambiguous gestures occasionally produced no-response outputs, which is actually the correct conservative behavior.
- **Spatial awareness** was effective: the model correctly identified whether the user was on the left or right side of the frame and chose the corresponding arm in most trials.
- **Smoothstep interpolation** produced noticeably more natural motion compared to a direct position-command baseline, eliminating the jerky start-stop behavior.
- **Inference latency (~4–6 sec)** is the primary bottleneck. This is a known limitation of running an 8B-parameter model on an edge device without quantization. Applying INT4 quantization or using a smaller VLM (e.g., 3B) could reduce latency significantly.
- Unlike OpenVLA or Diffusion Policy which require large-scale task-specific training data, **HARU requires zero robot-specific training** — the VLM's general world knowledge and instruction-following capability is leveraged directly.

---

## 8. Conclusion

This project presented HARU, a Physical AI robot that uses a Vision-Language Model as its sole decision-making engine for HRI. By treating the VLM as both a perceptual module and a motion planner, the system eliminates the need for hand-crafted rules or robot-specific training datasets.

The main contribution is a lightweight ROS2 pipeline — vision fusion, VLM-based trajectory generation, and smooth Dynamixel control — that enables zero-shot, gesture-responsive interaction on an edge device.

Future work includes reducing inference latency via model quantization, adding speech input, and extending the joint space to support more expressive full-body motions.

---

## 9. References

1. Kim, M. J., et al. **"OpenVLA: An Open-Source Vision-Language-Action Model."** arXiv:2406.09246 (2024).
2. Physical Intelligence. **"π0.5: A Vision-Language-Action Model for General Robot Control."** (2025).
3. Fu, Z., et al. **"HumanPlus: Humanoid Shadowing and Imitation from Humans."** arXiv:2406.10454 (2024).
4. Chi, C., et al. **"Diffusion Policy: Visuomotor Policy Learning via Action Diffusion."** RSS 2023.
5. Qwen Team. **"Qwen3 Technical Report."** Alibaba Group (2025).
