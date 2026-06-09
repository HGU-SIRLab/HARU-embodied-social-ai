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

## 3. Task and Method

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

## 4. Experiments

### 4.1 Setup
| Component | Spec |
|---|---|
| Platform | NVIDIA Jetson (aarch64, JetPack) |
| VLM | Qwen3-VL-8B-Instruct (float16) |
| Framework | ROS2, PyTorch 2.4/2.5 |
| Cameras | Intel RealSense D435 + Logitech C270 |
| Motors | Dynamixel Protocol 2.0 (×9) |

### 4.2 Evaluated Scenarios
The following interaction scenarios were tested:

| Scenario | Expected HARU Response |
|---|---|
| User waves hand directly at robot | Both arms wave back with greeting speech |
| User waves from the left side | Left arm preferentially raised |
| User waves from the right side | Right arm preferentially raised |
| User stands still / no gesture | Empty sequence returned, no movement |
| User makes eye contact only | Subtle head movement, short verbal response |

### 4.3 Latency Analysis
| Stage | Measured Time |
|---|---|
| Camera capture → ROS publish | ~33 ms (3 Hz) |
| Frame buffer fill (3 frames) | ~1 sec |
| Qwen3-VL-8B inference (Jetson) | ~3–5 sec |
| JSON parse → first motor command | < 5 ms |
| **Total end-to-end latency** | **~4–6 sec** |

---

## 5. Result Analysis

HARU successfully demonstrated **context-aware, autonomous motion generation** driven entirely by a VLM. Key observations:

- **Gesture recognition worked reliably** for clear, intentional gestures (waving, pointing). Subtle or ambiguous gestures occasionally produced no-response outputs, which is actually the correct conservative behavior.
- **Spatial awareness** was effective: the model correctly identified whether the user was on the left or right side of the frame and chose the corresponding arm in most trials.
- **Smoothstep interpolation** produced noticeably more natural motion compared to a direct position-command baseline, eliminating the jerky start-stop behavior.
- **Inference latency (~4–6 sec)** is the primary bottleneck. This is a known limitation of running an 8B-parameter model on an edge device without quantization. Applying INT4 quantization or using a smaller VLM (e.g., 3B) could reduce latency significantly.
- Unlike OpenVLA or Diffusion Policy which require large-scale task-specific training data, **HARU requires zero robot-specific training** — the VLM's general world knowledge and instruction-following capability is leveraged directly.

---

## 6. Conclusion

This project presented HARU, a Physical AI robot that uses a Vision-Language Model as its sole decision-making engine for HRI. By treating the VLM as both a perceptual module and a motion planner, the system eliminates the need for hand-crafted rules or robot-specific training datasets.

The main contribution is a lightweight ROS2 pipeline — vision fusion, VLM-based trajectory generation, and smooth Dynamixel control — that enables zero-shot, gesture-responsive interaction on an edge device.

Future work includes reducing inference latency via model quantization, adding speech input, and extending the joint space to support more expressive full-body motions.

---

## 7. References

1. Kim, M. J., et al. **"OpenVLA: An Open-Source Vision-Language-Action Model."** arXiv:2406.09246 (2024).
2. Physical Intelligence. **"π0.5: A Vision-Language-Action Model for General Robot Control."** (2025).
3. Fu, Z., et al. **"HumanPlus: Humanoid Shadowing and Imitation from Humans."** arXiv:2406.10454 (2024).
4. Chi, C., et al. **"Diffusion Policy: Visuomotor Policy Learning via Action Diffusion."** RSS 2023.
5. Qwen Team. **"Qwen3 Technical Report."** Alibaba Group (2025).
