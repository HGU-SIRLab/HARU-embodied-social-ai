# HARU — Embodied Social AI for Proactive Human-Robot Interaction

<p align="center">
  <img src="https://img.shields.io/badge/ROS2-Humble-blue" />
  <img src="https://img.shields.io/badge/Jetson-AGX_Orin_64GB-green" />
  <img src="https://img.shields.io/badge/Model-Gemma_4_12B-orange" />
  <img src="https://img.shields.io/badge/Phase-5_(QLoRA_+_HITL)-purple" />
</p>

**Author:** Cho Hyeongmin | Handong Global University, SIRLab  
**Research:** M.S. Thesis — Embodied Social AI for Proactive HRI

---

## Overview

**HARU** is a social companion robot that goes beyond mechanical command execution. It fuses **Physical AI** and **Human-Robot Interaction (HRI)** into a single embodied system capable of empathetic, proactive interaction with people of all ages.

Rather than reacting to explicit commands, HARU *observes* the human's facial expressions, body language, and voice tone — then *initiates* conversation first: *"Are you feeling sad because of something that happened?"*

The system is built on five core principles:

| Principle | Description |
|-----------|-------------|
| **Pre-trained Foundation** | Gemma 4 12B provides universal social common sense out of the box |
| **SWM + Adaptive ToM** | Social World Model + 6-stage Theory of Mind for deep context understanding |
| **Proactive HRI** | Robot initiates interaction without waiting for commands |
| **Human-in-the-Loop** | Human corrections are captured and used to improve behavior |
| **Episodic LoRA** | Daily interactions are accumulated as lightweight LoRA adapters, preventing catastrophic forgetting |

---

## Architecture — Hierarchical Dual-System

Inspired by Kahneman's System 1 / System 2 cognitive theory:

```
[RealSense SR300]──┐
                   ├──▶  haru_vision  ──▶  /haru_vision/compressed
[Logitech C270] ───┘         3Hz, 896×448 JPEG

[C270 Microphone] ──▶  haru_audio  ──▶  /haru_audio/raw
                         VAD, 16kHz              (Float32MultiArray)

                    ┌────────────────────────────────────────┐
                    │         haru_brain  (System 2)         │
                    │   Gemma 4 12B Unified  (~45-50s/turn)  │
                    │   MindPower 6-stage ToM                │
                    │   Social World Model (SWM)             │
                    │   Session Memory (50 pairs, disk)      │
                    │   PEFT LoRA adapter auto-load          │
                    └────────────┬───────────────────────────┘
                                 │ /haru_vla_raw (JSON)
                      ┌──────────▼──────────┐
                      │  [HITL mode]        │
                      │  haru_logger        │──▶  data/episodes/
                      │  Episode collection │
                      │  Kinesthetic teach  │
                      └──────────┬──────────┘
                                 │ /haru_system1_command
                    ┌────────────▼───────────────────────────┐
                    │        haru_action  (System 1)         │
                    │   50Hz Smoothstep interpolation        │
                    │   9 position joints + 2 wheel drives   │
                    └────────────┬───────────────────────────┘
                                 │
                    [Dynamixel U2D2  /dev/ttyACM0]
                    ID 3~12 (Protocol 2.0, 57600 baud)
```

### Why Dual-System?

OpenVLA-style single end-to-end models were evaluated first but rejected due to:
- **Language collapse** — discrete action tokenization corrupts the language generation space (ŸŸŸŸ phenomenon)
- **15-second latency** — autoregressive action token generation is incompatible with real-time HRI
- **Structural bottleneck** — simultaneous conversation + gesture generation is architecturally impossible

The Dual-System separates *deliberative reasoning* (System 2, slow but deep) from *reactive execution* (System 1, fast and smooth), achieving both linguistic quality and physical responsiveness.

---

## Key Features

### System 2 — Gemma 4 12B Unified
- **Native multimodal input**: raw pixels + 16kHz audio in a single encoder-free decoder
- **MindPower ToM**: 6-stage reasoning — Perception → Belief → Desire → Intention → Decision → Action
- **Session memory**: conversation history persisted across sessions (`data/memory/swm_history.json`)
- **Episodic LoRA**: PEFT adapter auto-loaded at startup from `data/adapters/`

### System 1 — haru_action
- **50Hz control loop** with Smoothstep (`3t² − 2t³`) interpolation
- **9 position-controlled joints** + **2 velocity-controlled wheels**
- **Kinesthetic mode**: hardware torque ON/OFF for direct physical teaching
- Publishes joint states at 10Hz (`/haru_joints/state`) for HITL capture

### HITL Pipeline
- **[A] Accept** — robot executes VLA proposal as-is, saved as positive example
- **[C] Correct** — two correction modes:
  - **[D] Direct (Kinesthetic Teaching)** — torque OFF → physically move robot → Enter → encoder auto-captured
  - **[M] Manual** — type joint values directly (Enter = keep current)
- **[S] Skip** — robot executes but data not logged
- **[E] End** — save episode with metadata

### QLoRA Training (`scripts/train_lora.py`)
- `HaruEpisodeDataset` — PyTorch Dataset with pre-tokenized samples and accurate loss masking
- Prompt tokens masked with `labels[:prompt_len] = -100`
- `gradient_checkpointing` with `use_reentrant=False` (Gemma 4 KV-sharing bug workaround)
- Saves to `data/adapters/adapter_YYYYMMDD_HHMMSS/`; auto-loaded on next brain_node start

---

## Hardware

| Component | Model | Role | Connection |
|-----------|-------|------|------------|
| Main Computer | NVIDIA Jetson AGX Orin 64GB | Full system | — |
| Face Camera | Intel RealSense SR300 | Face & gaze | USB → `/dev/video0` |
| Body Camera | Logitech C270 | Body & gestures | USB → `/dev/video4` |
| Microphone | Logitech C270 USB Audio | Voice tone & intonation | PulseAudio |
| Motor Adapter | ROBOTIS U2D2 | USB↔TTL bridge | USB → `/dev/ttyACM0` |
| Servo Motors | Dynamixel XM/XH series ×11 | 9 joints + 2 wheels | TTL daisy chain |

### Joint Map

| Joint | ID | Range | Role |
|-------|----|-------|------|
| r_arm_pitch | 3 | 1024–2451 | Right arm pitch |
| l_arm_pitch | 4 | 37–1542 | Left arm pitch |
| r_shoulder_roll | 5 | 1000–2050 | Right shoulder roll |
| r_elbow_pitch | 6 | 2047–3062 | Right elbow |
| l_shoulder_roll | 7 | 1047–2056 | Left shoulder roll |
| l_elbow_pitch | 8 | 1021–2007 | Left elbow |
| head_pan | 10 | 1043–3071 | Head yaw (left/right) |
| head_tilt | 11 | 1500–3086 | Head pitch (nod) |
| head_roll | 12 | 1630–2452 | Head roll (tilt) |
| right_wheel | 1 | −300–300 | Right drive wheel |
| left_wheel | 2 | −300–300 | Left drive wheel |

### Expression IDs

| ID | Expression | ID | Expression |
|----|------------|----|------------|
| 0 | neutral | 4 | surprise |
| 1 | joy | 5 | empathy |
| 2 | sadness | 6 | thinking |
| 3 | curiosity | 7 | concern |

---

## Software Requirements

| Software | Version | Notes |
|----------|---------|-------|
| JetPack SDK | 6.2.2 | Ubuntu 22.04, CUDA 12.6 |
| ROS2 | Humble | `apt install ros-humble-desktop` |
| Python | 3.10 | Included with JetPack |
| PyTorch | 2.5.0 (Jetson wheel) | **Do NOT use `pip install torch`** |
| transformers | ≥ 5.12.0 | Gemma 4 `Gemma4UnifiedForConditionalGeneration` |
| peft | ≥ 0.19.1 | LoRA adapter loading |
| trl | ≥ 1.6.0 | QLoRA training utilities |
| sounddevice | ≥ 0.5.5 | Microphone capture |
| scipy | ≥ 1.15.3 | Audio resampling |
| dynamixel-sdk | — | Motor control |
| opencv-python | — | Camera capture |

> ⚠️ **Jetson PyTorch**: Always use the NVIDIA-provided wheel. Standard PyPI builds have no CUDA support on aarch64.

> ⚠️ **bitsandbytes 4-bit**: Incompatible with Gemma 4 Unified architecture (confirmed on bitsandbytes 0.49.2). The system loads in **bf16 (~22GB)**. Do not pass `load_in_4bit=True`.

---

## Installation

### 1. ROS2 Humble

```bash
sudo apt update && sudo apt install -y ros-humble-desktop
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. Python Virtual Environment

```bash
cd ~/robot_brain_workspace
python3 -m venv haru_vla_env
source haru_vla_env/bin/activate
```

### 3. PyTorch (Jetson wheel)

```bash
# Use the Jetson-specific wheel — DO NOT use pip install torch
pip install torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
```

### 4. Python Packages

```bash
pip install transformers>=5.12.0 peft>=0.19.1 trl>=1.6.0 \
    accelerate safetensors huggingface-hub \
    sounddevice scipy opencv-python Pillow numpy \
    dynamixel-sdk
```

### 5. Build ROS2 Workspace

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 6. Model Download

On first run, `haru_brain` downloads `google/gemma-4-12B-it` (~23GB) from HuggingFace automatically. To pre-download:

```bash
source haru_vla_env/bin/activate
python3 -c "
from transformers import Gemma4UnifiedForConditionalGeneration, AutoProcessor
AutoProcessor.from_pretrained('google/gemma-4-12B-it', trust_remote_code=True)
Gemma4UnifiedForConditionalGeneration.from_pretrained(
    'google/gemma-4-12B-it', dtype='bfloat16', trust_remote_code=True)
"
```

---

## Running HARU

```bash
# Required in every terminal
source ~/.bashrc   # loads ROS2 + aliases
```

### Normal Operation (no data collection)

```bash
haru-ws && haru-all
# or individually:
haru-vision   # camera node
haru-brain    # Gemma 4 inference (~24s to load)
haru-action   # motor control
haru-audio    # microphone VAD
```

### HITL Data Collection Mode

```bash
haru-ws && haru-hitl
```

Terminal controls:
- **[N]** Start new episode
- **[A]** Accept VLA proposal → log + execute
- **[C]** Correct → choose **[D]** kinesthetic or **[M]** manual
- **[S]** Skip this step (execute but don't log)
- **[E]** End & save episode
- **[Q]** Cancel episode (delete data)

### QLoRA Training (after collecting episodes)

```bash
source haru_vla_env/bin/activate
cd ~/robot_brain_workspace

# Check data first (no model load)
python scripts/train_lora.py --dry-run

# Train (default: rank=16, epochs=3, lr=2e-4)
python scripts/train_lora.py

# Custom
python scripts/train_lora.py --epochs 5 --rank 32 --lr 1e-4
```

Adapter is saved to `data/adapters/adapter_YYYYMMDD_HHMMSS/` and **auto-loaded** on the next `haru-brain` start.

---

## Repository Structure

```
HARU-embodied-social-ai/
├── src/
│   ├── haru_vision/          # Camera node (dual-cam, 3Hz)
│   │   └── haru_vision/vision_node.py
│   ├── haru_brain/           # System 2 — Gemma 4 12B
│   │   └── haru_brain/
│   │       ├── brain_node.py
│   │       ├── gemma4_inference.py   # Inference + LoRA auto-load
│   │       ├── tom_prompt.py         # MindPower ToM + SWM builder
│   │       ├── session_memory.py     # Cross-session persistence
│   │       └── adapter_manager.py    # LoRA adapter selector
│   ├── haru_audio/           # Microphone node (VAD, 16kHz)
│   │   └── haru_audio/audio_node.py
│   ├── haru_action/          # System 1 — 50Hz motor control
│   │   └── haru_action/action_node.py
│   └── haru_logger/          # HITL episode logger
│       └── haru_logger/
│           ├── hitl_node.py          # Interactive correction UI
│           └── episode_writer.py     # 12-DoF normalized NPZ writer
├── scripts/
│   └── train_lora.py         # QLoRA fine-tuning pipeline
├── data/                     # ← git-ignored (episodes, adapters, memory)
│   ├── episodes/             # HITL collected steps (step_XXXX.npz)
│   ├── adapters/             # Trained LoRA adapters
│   └── memory/               # Session SWM history
├── launch_vla.sh             # Launch script
├── HARU_PROJECT_CONTEXT.md   # Full architecture reference
├── HARU_RUN_COMMANDS.txt     # All commands & usage guide (Korean)
└── HARU_RESEARCH_GOALS.txt   # M.S. thesis research goals & gap analysis
```

---

## ROS2 Topic Map

| Topic | Type | Direction |
|-------|------|-----------|
| `/haru_vision/compressed` | `CompressedImage` | vision → brain |
| `/haru_audio/raw` | `Float32MultiArray` (16kHz PCM) | audio → brain |
| `/haru_audio/vad` | `Bool` | audio → monitor |
| `/haru_vla_raw` | `String` (JSON) | brain → logger/action |
| `/haru_system1_command` | `String` (JSON) | logger → action *(HITL only)* |
| `/haru_expression` | `Int32` | brain/logger → display |
| `/haru_speech` | `String` | brain → TTS |
| `/haru_joints/state` | `Float32MultiArray` (9-dim, 10Hz) | action → logger |
| `/haru_joints/torque` | `Bool` | logger → action *(kinesthetic)* |

---

## Data Format

Each episode step is stored as a compressed NumPy archive:

```python
# step_XXXX.npz
{
  "image":                (448, 896, 3)  uint8    # dual-camera frame
  "action":               (12,)          float32  # corrected pose, normalized [-1, 1]
  "action_vla":           (12,)          float32  # original VLA proposal
  "is_corrected":         bool                    # whether human corrected this step
  "language_instruction": bytes                   # task description
  "speech_text":          bytes                   # what robot said
  "emotion":              bytes                   # detected emotion label
}
```

12-DoF order: `expression_id, head_tilt, head_pan, head_roll, r_arm_pitch, l_arm_pitch, r_shoulder_roll, r_elbow_pitch, l_shoulder_roll, l_elbow_pitch, right_wheel, left_wheel`

---

## Research Contributions

1. **Hierarchical Dual-System for Social HRI** — separating deliberative VLM reasoning (System 2) from reactive motor execution (System 1) enables both linguistic quality and physical responsiveness on an edge device

2. **MindPower ToM + SWM** — 6-stage Theory of Mind (Perception→Belief→Desire→Intention→Decision→Action) with cross-session Social World Model persistence

3. **Kinesthetic Teaching via HITL** — physical pose correction through direct manipulation with torque-off, encoder-capture pipeline; no need to know joint values

4. **Episodic LoRA for Catastrophic Forgetting Prevention** — each interaction session is distilled into a LoRA adapter, keeping the base model identity intact while accumulating personalized behavior

---

## Limitations & Future Work

| Limitation | Status | Plan |
|------------|--------|------|
| ~45–50s inference latency | Known (Jetson + Gemma 4 12B bf16) | VAD-triggered inference, lighter fallback model |
| No TTS output | `/haru_speech` topic ready | Piper TTS integration |
| No expression display | `/haru_expression` topic ready | OLED / LED matrix node |
| Single adapter (no context routing) | Latest adapter only | S-LoRA multi-adapter serving (Phase 6) |
| SFT only (no reward model) | HITL-SFT | DPO after sufficient data collection |
| System 1 = Smoothstep only | Adequate for social gestures | ACT / Diffusion Policy (Phase 6) |

---

## References

1. Kahneman, D. *Thinking, Fast and Slow.* Farrar, Straus and Giroux, 2011.
2. Google DeepMind. **"Gemma 4 Technical Report."** (2026).
3. Hu, E., et al. **"LoRA: Low-Rank Adaptation of Large Language Models."** ICLR 2022.
4. Chi, C., et al. **"Diffusion Policy: Visuomotor Policy Learning via Action Diffusion."** RSS 2023.
5. Zhao, T., et al. **"Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)."** RSS 2023.
6. Kim, M. J., et al. **"OpenVLA: An Open-Source Vision-Language-Action Model."** arXiv:2406.09246 (2024).
7. Dettmers, T., et al. **"QLoRA: Efficient Finetuning of Quantized LLMs."** NeurIPS 2023.
