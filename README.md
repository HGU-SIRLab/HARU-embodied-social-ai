# HARU — Embodied Social AI for Proactive Human-Robot Interaction

<p align="center">
  <img src="https://img.shields.io/badge/ROS2-Humble-blue" />
  <img src="https://img.shields.io/badge/Jetson-AGX_Orin_64GB-green" />
  <img src="https://img.shields.io/badge/Model-Gemma_4_12B-orange" />
  <img src="https://img.shields.io/badge/Phase-6.5.5_(완료)-purple" />
  <img src="https://img.shields.io/badge/Inference-vLLM_W4A16_19tok%2Fs-red" />
  <img src="https://img.shields.io/badge/Speech-0.65s_warm_(69×)-brightgreen" />
</p>

**Author:** Cho Hyeongmin (조형민) | Handong Global University, SIRLab  
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

## Architecture — Hierarchical Triple-System

Inspired by Kahneman's System 1 / System 2 cognitive theory, extended with a perceptual attention layer (System 3):

```
[RealSense SR300]──┐
                   ├──▶  haru_vision  ──▶  /haru_vision/compressed
[Logitech C270] ───┘         3Hz, 896×448 JPEG
                                  │
                    ┌─────────────▼────────────────┐
                    │    haru_attention (System 3)  │  ← 항상 켜짐 (CPU)
                    │  OpenCV Haar + 5-State FSM    │
                    │  EMPTY/APPEARED/CONVERSING    │
                    │  PRESENT_SILENT/LONG_IDLE     │
                    └──────────────┬───────────────┘
                                   │ /haru_attention/event (JSON)

[C270 Microphone] ──▶  haru_audio  ──▶  /haru_audio/raw
                         VAD, 16kHz              (Float32MultiArray)
                                   │ /haru_audio/vad (Bool)
                    ┌──────────────▼─────────────────────────────────┐
                    │          haru_brain  (System 2)                │
                    │  Gemma 4 12B Unified                           │
                    │  ┌────────────────────────────────────────┐    │
                    │  │  3-Tier 추론 경로 자동 선택             │    │
                    │  │  1. TRT-LLM W4A16  (~6-10s/turn) ★    │    │
                    │  │  2. AutoRound W4A16 (~10-20s/turn) ✅  │    │
                    │  │  3. HF bf16 GPU    (~15-30s/turn)      │    │
                    │  └────────────────────────────────────────┘    │
                    │  MindPower 6-stage ToM                         │
                    │  Social World Model (SWM, 50 pairs on disk)    │
                    │  PEFT LoRA adapter auto-load                   │
                    └────────────┬───────────────────────────────────┘
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

### Why Triple-System?

OpenVLA-style single end-to-end models were evaluated first but rejected due to:
- **Language collapse** — discrete action tokenization corrupts the language generation space
- **15-second latency** — incompatible with real-time HRI
- **Structural bottleneck** — simultaneous conversation + gesture generation is architecturally impossible

The Triple-System separates *perceptual attention* (System 3, always-on CPU), *deliberative reasoning* (System 2, event-driven GPU), and *reactive execution* (System 1, 50Hz), achieving both linguistic quality and physical responsiveness on an edge device.

---

## Phase 6.4 — Action Early Callback + Abbreviated Keys: 7.4s total ✅ (2026-06-26)

### Current Inference Architecture

| Tier | Method | Size | Latency | Status |
|------|--------|------|---------|--------|
| **1 (Active)** | **vLLM W4A16 + Marlin INT4 + CUDAGraph** | **~7.4 GB** | **19 tok/s · speech ~2s warm · full 7.4s** | **✅ Running** |
| 2 | auto_round W4A16 direct load | ~7.4 GB | ~10–20s | ✅ fallback |
| 3 | HF Transformers bf16 | ~22 GB | ~15–30s | ✅ fallback |

**Key optimizations (cumulative):**

| Phase | Optimization | Speedup | Result |
|-------|-------------|---------|--------|
| 5.8 | vLLM W4A16 Marlin INT4 + CUDAGraph | 7× tok/s | 2.8 → 19 tok/s |
| 6.3 | Streaming `_extract_speech_field` | early speech | speech 0.65s warm (cold 2.65s) |
| 6.3 | `--enable-prefix-caching` | TTFT cached | system prompt 383 tok KV cached |
| 6.3 | IMG_SIZE 448→336 | 44% fewer vision tokens | ~0.6s prefill savings |
| 6.3 | SWM window 4→2 + response compression | ~300 tok saved/turn | — |
| **6.4** | **Abbreviated action JSON keys (32 tok saved)** | **~2.2s** | **9.6s → 7.4s full inference** |
| **6.4** | **`action_ready_cb` early body movement** | **~0.5s** | body moves 0.5s before JSON complete |
| **Total** | **HF bf16 ~45s → full inference 7.4s** | **≥69×** | **✅ 목표 달성** |

### Full Pipeline Timing (7/7 nodes, haru_all, Phase 6.4)

| Event | Time from trigger | Notes |
|-------|------------------|-------|
| speech (early, warm) | **~2.0s** avg | `_extract_speech_field` streaming |
| expression (early, warm) | **~2.7s** avg | `_extract_expression_id` streaming |
| **action (early, warm)** | **~7.9s** avg | `_extract_action_dict` streaming (new 6.4) |
| full inference (11-joint abbreviated) | **~7.4s** avg | 73 tok action JSON vs old 105 tok |
| TTS audio output | +400ms after speech field | edge-tts ko-KR-SunHiNeural |

### Action JSON Format (Phase 6.4 abbreviated keys)

```json
{"speech":"...","emotion":"joy","expression_id":1,
 "action":{"ht":2048,"hp":2057,"hr":2041,"rap":1738,"rsr":1525,"rep":2555,
            "lap":790,"lsr":1552,"lep":1514,"rw":0,"lw":0},"duration":2.5}
```
Keys: `ht`=head_tilt `hp`=head_pan `hr`=head_roll | `rap`=r_arm_pitch `rsr`=r_shoulder_roll `rep`=r_elbow_pitch | `lap`=l_arm_pitch `lsr`=l_shoulder_roll `lep`=l_elbow_pitch | `rw`=right_wheel `lw`=left_wheel

### Starting the System

```bash
# Step 1: Start vLLM inference server (once, stays up)
bash scripts/run_vllm_server.sh
# Verify: curl http://localhost:8000/v1/models

# Step 2: Start all 7 nodes
bash launch_vla.sh
```

### haru_all — 7-Node Pipeline (Phase 6.2+)

```
vision_node      — 3Hz dual-cam 336×336 JPEG
attention_node   — 5-State FSM, face/motion/VAD
audio_node       — 16kHz VAD
brain_node       — Gemma 4 12B, streaming, prefix caching
action_node      — 50Hz Smoothstep, 9 joints + 2 wheels
tts_node         — edge-tts ko-KR, 400ms, mpg123 hw:3,0
expression_node  — Robot-display-HRI pygame FULLSCREEN, 14 emotions, DISPLAY=:0
```

### Phase 5.7 Integration Test Results (8/8 ✅, 2026-06-25)

| Checkpoint | Result |
|-----------|--------|
| vLLM container port 8000 response | ✅ |
| `Gemma4TRTLLMInference.try_connect()` | ✅ True, model="gemma4" |
| brain_node auto TRT-LLM mode | ✅ |
| `/haru_vision/compressed` receipt | ✅ |
| `/haru_attention/event` trigger processing | ✅ |
| Vision + text joint inference | ✅ |
| `/haru_speech` publish | ✅ "어머, 안녕하세요! 당신이 나타나니 정말 기뻐요." |
| `/haru_expression` + `/haru_vla_raw` publish | ✅ expr=1 (joy), head=(2300,2057,2041) |

### auto_round W4A16 Key Technical Notes

Gemma 4 Unified has **heterogeneous attention**:
- `sliding_attention` layers (every 5): `head_dim=256`, local RoPE
- `full_attention` layers (every 6th): `global_head_dim=512`, global RoPE

**Fix 1 — RoPE dimension mismatch:** `apply_rotary_pos_emb` monkey-patch with `rot_dim = min(x.shape[-1], cos.shape[-1])`.

**Fix 2 — Jetson PyTorch 2.5 `set_submodule` bug (2026-06-23):**
`set_submodule` uses `type(mod) is not torch.nn.Module` (exact check), causing AttributeError on all nn.Module subclasses → auto_round `set_module()` silently ignores it → QuantLinear never installed → original weights destroyed.

**Fix:** `_patched_set_module` (direct `getattr`/`setattr`) in `quantize_gemma4_autoround.py`. Verified: 328 qweight, 1333 keys, 2 shards.

---

## Key Features

### System 3 — haru_attention (항상 켜짐, CPU)
- **5-State FSM**: EMPTY → APPEARED → CONVERSING ↔ PRESENT_SILENT → LONG_IDLE
- **OpenCV Haar Cascade** 얼굴 감지 (~5ms/frame), 프레임 차분 모션 감지
- **이벤트 드리븐 트리거**: Gemma 4를 '사회적으로 의미 있는 상황'에서만 깨움
- **VAD 통합**: off-camera 발화 시 CONVERSING 전환 (race condition 수정됨)

### System 2 — haru_brain (Gemma 4 12B VLM)
- **Native multimodal**: raw pixels + 16kHz audio in a single pass
- **MindPower ToM**: 6-stage reasoning — Perception → Belief → Desire → Intention → Decision → Action
- **Silence selection**: `speech: ""` → gesture-only response without speech
- **Streaming early callbacks**: `speech_ready_cb` (~2s warm), `expression_ready_cb` (~2.7s), `action_ready_cb` (~7.9s, body moves before JSON complete)
- **Prefix caching**: `--enable-prefix-caching` — system prompt 383 tokens KV-cached, TTFT ~0.3s warm
- **Abbreviated action keys**: 32 tokens saved vs verbose names — 9.6s→7.4s full inference
- **Session memory**: SWM (window=2 pairs) persisted cross-session (`data/memory/swm_history.json`)
- **Episodic LoRA**: PEFT adapter auto-loaded at startup from `data/adapters/`
- **11-joint output**: all 9 position joints + 2 wheel drives in every inference response

### System 1 — haru_action
- **50Hz control loop** with Smoothstep (`3t² − 2t³`) interpolation
- **9 position-controlled joints** + **2 velocity-controlled wheels**
- **Kinesthetic mode**: hardware torque ON/OFF for direct physical teaching

### HITL Pipeline
- **[A] Accept** — robot executes VLA proposal as-is, saved as positive example
- **[C] Correct** — two modes:
  - **[D] Direct (Kinesthetic Teaching)** — torque OFF → physically move → Enter → encoder captured
  - **[M] Manual** — type joint values directly
- **[S] Skip** — execute but don't log
- **[E] End** — save episode with metadata including FSM state + context

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

| Joint (abbr) | Full Name | ID | Range | Neutral |
|-------------|-----------|-----|-------|---------|
| **rap** | r_arm_pitch | 3 | 1024–2451 | 1738 |
| **lap** | l_arm_pitch | 4 | 37–1542 | 790 |
| **rsr** | r_shoulder_roll | 5 | 1000–2050 | 1525 |
| **rep** | r_elbow_pitch | 6 | 2047–3062 | 2555 |
| **lsr** | l_shoulder_roll | 7 | 1047–2056 | 1552 |
| **lep** | l_elbow_pitch | 8 | 1021–2007 | 1514 |
| **hp** | head_pan | 10 | 1043–3071 | 2057 |
| **ht** | head_tilt | 11 | 1500–3086 | 2048 |
| **hr** | head_roll | 12 | 1630–2452 | 2041 |
| **rw** | right_wheel | 1 | −300–300 | 0 |
| **lw** | left_wheel | 2 | −300–300 | 0 |

### Expression IDs

HARU uses **Robot-display-HRI** (https://github.com/HGU-SIRLab/Robot-display-HRI) as the pygame face renderer — 14 anime-style emotions, fullscreen 1024×600.

| HARU ID | Brain emotion | Robot-display-HRI key |
|---------|--------------|----------------------|
| 0 | neutral | NEUTRAL |
| 1 | joy | HAPPY |
| 2 | sadness | SAD |
| 3 | curiosity | THINKING |
| 4 | surprise | SURPRISED |
| 5 | empathy | TENDER |
| 6 | thinking | THINKING |
| 7 | concern | SCARED |

Additional renderer-internal emotions (not mapped from brain output): `EXCITED`, `ANGRY`, `LISTENING`, `CLOSE`, `SCANNING`, `SLEEPY`, `WAKE`

---

## Software Requirements

| Software | Version | Notes |
|----------|---------|-------|
| JetPack SDK | 6.2.2 | Ubuntu 22.04, CUDA 12.6, L4T 36.5.0 |
| ROS2 | Humble | `apt install ros-humble-desktop` |
| Python | 3.10 | Included with JetPack |
| PyTorch | **2.5.0a0+nv24.08** (Jetson wheel) | **절대 `pip install torch` 금지** |
| transformers | ≥ 5.12.1 | `Gemma4UnifiedForConditionalGeneration` |
| auto_round | 0.13.1 | W4A16 weight quantization |
| peft | ≥ 0.19.1 | LoRA adapter loading |
| trl | ≥ 1.6.0 | QLoRA training |
| sounddevice | ≥ 0.5.5 | Microphone capture |
| scipy | ≥ 1.15.3 | Audio resampling |
| dynamixel-sdk | — | Motor control |
| opencv-python | — | Camera capture + face detection |

> ⚠️ **Jetson PyTorch**: Always use the NVIDIA-provided wheel (`torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl`). Standard PyPI builds target CUDA 13.0+ and will not activate the GPU on Jetson (CUDA 12.6).

> ⚠️ **bitsandbytes 4-bit**: Incompatible with SM87 (Jetson AGX Orin). Use auto_round W4A16 or TRT-LLM for 4-bit quantization instead.

> ℹ️ **setuptools 81.0.0**: Breaks `colcon --symlink-install` editable installs. `haru_brain` and `haru_attention` use a manual install structure (symlinks). Source changes take effect immediately without rebuild.

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

### 3. PyTorch (Jetson wheel — MANDATORY)

```bash
# DO NOT use pip install torch
pip install torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
```

### 4. Python Packages

```bash
pip install transformers>=5.12.0 peft>=0.19.1 trl>=1.6.0 \
    accelerate safetensors huggingface-hub auto-round==0.13.1 \
    sounddevice scipy opencv-python Pillow numpy \
    dynamixel-sdk
```

### 5. Build ROS2 Workspace

```bash
source /opt/ros/humble/setup.bash
# Standard packages
colcon build --symlink-install \
    --packages-select haru_audio haru_vision haru_action haru_logger
source install/setup.bash
# haru_brain / haru_attention use manual install (symlinks already set up)
```

### 6. Model — First Run

On first run, `haru_brain` loads from HuggingFace cache (pre-downloaded ~23GB). To pre-download:

```bash
source haru_vla_env/bin/activate
python3 -c "
from transformers import Gemma4UnifiedForConditionalGeneration, AutoProcessor
AutoProcessor.from_pretrained('google/gemma-4-12B-it', trust_remote_code=True)
Gemma4UnifiedForConditionalGeneration.from_pretrained(
    'google/gemma-4-12B-it', dtype='bfloat16', trust_remote_code=True)
"
```

### 7. Start Inference Server (vLLM Container)

```bash
# Start Gemma4 vision server on port 8000
bash scripts/run_vllm_server.sh

# Verify
curl http://localhost:8000/v1/models

# Stop
docker stop haru_vllm_server
```

brain_node automatically connects when port 8000 is available (log: `"✅ TRT-LLM 서버 연결 성공"`).

### 8. W4A16 Quantization (run once, already done)

```bash
source haru_vla_env/bin/activate
nohup python3 scripts/quantize_gemma4_autoround.py 2>&1 | tee /tmp/quantize.log &

# Monitor progress
tail -f /tmp/quantize.log

# Verify completion
ls data/gemma4_autoround_w4a16/   # config.json + *.safetensors
```

After completion, `haru-brain` automatically selects the W4A16 path on next start.

---

## Running HARU

```bash
# Required in every terminal
source ~/.bashrc
```

### Step 1: Start Inference Server

```bash
# Start Gemma4 VLM server on port 8000 (run once, stays up)
bash scripts/run_vllm_server.sh
# Check: curl http://localhost:8000/v1/models
```

### Step 2: Normal Operation (Triple-System)

```bash
haru-ws && haru-all
# Starts: vision + attention + brain + audio + action
# brain_node auto-detects port 8000 → TRT-LLM mode (W4A16 VLM server)
```

### HITL Data Collection Mode

```bash
haru-ws && haru-hitl
```

Terminal controls:
- **[N]** Start new episode
- **[A]** Accept VLA proposal → log + execute
- **[C]** Correct → **[D]** kinesthetic or **[M]** manual
- **[S]** Skip (execute but don't log)
- **[E]** End & save episode
- **[Q]** Cancel episode

### QLoRA Training (after collecting episodes)

```bash
source haru_vla_env/bin/activate

# Check data (no model load)
python scripts/train_lora.py --dry-run

# Train (rank=16, epochs=3, lr=2e-4)
python scripts/train_lora.py

# Custom
python scripts/train_lora.py --epochs 5 --rank 32 --lr 1e-4
```

Adapter is saved to `data/adapters/adapter_YYYYMMDD_HHMMSS/` and **auto-loaded** on the next `haru-brain` start.

### Test Quantized Model

```bash
source haru_vla_env/bin/activate
python scripts/test_autoround.py
# → 20+ checks: load, timing, JSON structure, edge cases, brain_node integration
```

---

## Repository Structure

```
robot_brain_workspace/
├── src/
│   ├── haru_vision/              # Camera node (dual-cam, 3Hz, 896×448 JPEG)
│   │   └── haru_vision/vision_node.py
│   ├── haru_attention/           # System 3 — perceptual attention (CPU-only)
│   │   └── haru_attention/attention_node.py  # 5-State FSM + face/motion/VAD
│   ├── haru_brain/               # System 2 — Gemma 4 12B
│   │   └── haru_brain/
│   │       ├── brain_node.py                  # 3-tier auto-select + attention event sub
│   │       ├── gemma4_inference.py            # HF bf16 baseline path
│   │       ├── gemma4_autoround_inference.py  # auto_round W4A16 path ★
│   │       ├── gemma4_trtllm_inference.py     # TRT-LLM high-speed path ★
│   │       ├── tom_prompt.py                  # MindPower ToM + SWM + silence rules
│   │       ├── session_memory.py              # Cross-session disk persistence
│   │       └── adapter_manager.py             # LoRA adapter selector (mtime)
│   ├── haru_audio/               # Microphone node (VAD, 16kHz, C270)
│   │   └── haru_audio/audio_node.py
│   ├── haru_action/              # System 1 — 50Hz motor control
│   │   └── haru_action/action_node.py
│   ├── haru_tts/                 # TTS node (edge-tts ko-KR, Phase 6.1)
│   │   └── haru_tts/tts_node.py
│   ├── haru_expression/          # Expression display node (Phase 6.5.5)
│   │   └── haru_expression/
│   │       ├── expression_node.py     # ROS2 wrapper: /haru_expression → emotion_queue
│   │       └── robot_face/            # Robot-display-HRI source (https://github.com/HGU-SIRLab/Robot-display-HRI)
│   │           ├── main.py            # RobotFaceApp + run_face_app(), FULLSCREEN 1024×600
│   │           ├── common_helpers.py  # Colors, gradient drawing helpers
│   │           └── emotions/          # 14 emotion modules (neutral/happy/sad/…)
│   └── haru_logger/              # HITL episode logger
│       └── haru_logger/
│           ├── hitl_node.py          # Interactive correction UI (FSM state display)
│           └── episode_writer.py     # 12-DoF NPZ + attention context writer
├── scripts/
│   ├── quantize_gemma4_autoround.py  # W4A16 quantization (RoPE patch included) ★
│   ├── test_autoround.py             # Full quantized model test suite ★
│   ├── test_gemma4.py                # bf16 baseline test
│   ├── train_lora.py                 # QLoRA fine-tuning pipeline
│   ├── gemma4_server.py              # ★ vLLM container FastAPI server (vision support)
│   ├── run_vllm_server.sh            # ★ Start haru_vllm_server container
│   ├── build_trtllm_docker.sh        # TRT-LLM Docker build (8-16h, one-time)
│   ├── convert_gemma4_trtllm.sh      # TRT-LLM engine conversion
│   ├── run_trtllm_server.sh          # TRT-LLM OpenAI-compatible server
│   └── convert_to_rlds.py            # Episode → RLDS format
├── data/                             # ← git-ignored
│   ├── episodes/                     # HITL collected steps (step_XXXX.npz, 9 keys)
│   ├── adapters/                     # Trained LoRA adapters (auto-loaded)
│   ├── memory/                       # SWM session history (swm_history.json)
│   ├── gemma4_autoround_w4a16/       # W4A16 quantized model (after quantization)
│   └── trtllm/                       # TRT-LLM engines (after Docker build)
├── haru_vla_env/                     # Python venv (git-ignored)
├── launch_vla.sh                     # Launch script
├── HARU_PROJECT_CONTEXT.md           # Full architecture reference (Korean)
├── HARU_RUN_COMMANDS.txt             # All commands & usage guide (Korean)
└── HARU_RESEARCH_GOALS.txt           # M.S. thesis goals & gap analysis (Korean)
```

---

## ROS2 Topic Map

| Topic | Type | Publisher → Subscriber |
|-------|------|------------------------|
| `/haru_vision/compressed` | `CompressedImage` | vision → attention, brain |
| `/haru_audio/raw` | `Float32MultiArray` (16kHz PCM) | audio → brain |
| `/haru_audio/vad` | `Bool` | audio → attention |
| `/haru_attention/event` | `String` (JSON) | attention → brain |
| `/haru_vla_raw` | `String` (JSON) | brain → logger/action |
| `/haru_system1_command` | `String` (JSON) | logger → action *(HITL only)* |
| `/haru_expression` | `Int32` | brain/logger → *(display node, Phase 6)* |
| `/haru_speech` | `String` | brain → *(TTS node, Phase 6)* |
| `/haru_joints/state` | `Float32MultiArray` (9-dim, 10Hz) | action → logger |
| `/haru_joints/torque` | `Bool` | logger → action *(kinesthetic)* |

---

## Data Format

Each HITL episode step is stored as a compressed NumPy archive:

```python
# data/episodes/episode_YYYYMMDD_HHMMSS/step_XXXX.npz
{
  "image":              (448, 896, 3)  uint8    # dual-camera frame
  "action":             (12,)          float32  # corrected pose, normalized [-1, 1]
  "action_vla":         (12,)          float32  # original VLA proposal
  "is_corrected":       bool                    # whether human corrected
  "language_instruction": bytes                 # task description
  "speech_text":        bytes                   # what robot said
  "emotion":            bytes                   # emotion label
  "attention_source":   bytes                   # FSM state (e.g. "CONVERSING")  ★Phase 5.5
  "attention_context":  bytes                   # situation context string        ★Phase 5.5
}
```

12-DoF order: `expression_id, head_tilt, head_pan, head_roll, r_arm_pitch, l_arm_pitch, r_shoulder_roll, r_elbow_pitch, l_shoulder_roll, l_elbow_pitch, right_wheel, left_wheel`

---

## Development Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **Phase 1** | Jetson 환경 구축 + 기본 추론 검증 | ✅ 완료 |
| **Phase 2** | ROS2 멀티노드 통합 | ✅ 완료 |
| **Phase 3** | HITL 에피소드 로깅 파이프라인 | ✅ 완료 |
| **Phase 4** | Gemma 4 12B + MindPower ToM + SWM | ✅ 완료 2026-06-18 |
| **Phase 4.5** | haru_audio + VAD | ✅ 완료 2026-06-18 |
| **Phase 5** | QLoRA 파이프라인 + SWM + Kinesthetic | ✅ 완료 2026-06-19 |
| **Phase 5.5** | **Triple-System**: haru_attention + 이벤트 드리븐 | ✅ 완료 2026-06-22 |
| **Phase 5.6** | GPU 복구 + auto_round W4A16 + TRT-LLM 경로 | ✅ **완료 2026-06-23** |
| **Phase 5.7** | vLLM 컨테이너 VLM 서버 + 비전 지원 + brain_node 통합 | ✅ **완료 2026-06-25** |
| **Phase 5.8** | 추론 속도 가속 (2.8 tok/s → 19 tok/s, 7×) | ✅ **완료 2026-06-26** |
| **Phase 6.1** | TTS 노드 (edge-tts ko-KR-SunHiNeural, 400ms, mpg123) | ✅ **완료 2026-06-26** |
| **Phase 6.2** | 표정 디스플레이 노드 (pygame 8 emotions, haru_all 7/7) | ✅ **완료 2026-06-26** |
| **Phase 6.3** | Streaming + prefix caching: speech ~2s warm (69×) | ✅ **완료 2026-06-26** |
| **Phase 6.4** | action_ready_cb + 약어 키: full inference 9.6s→7.4s | ✅ **완료 2026-06-26** |
| **Phase 6.5** | HITL 에피소드 수집 + 첫 LoRA 어댑터 (6.38s 중앙값) | ✅ **완료** |
| **Phase 6.5.5** | **Robot-display-HRI pygame 통합** — 14 anime 감정, FULLSCREEN 1024×600 | ✅ **완료 2026-07-01** |
| **Phase 6.6** | HITL 에피소드 수집 심화 + 첫 LoRA 어댑터 검증 | 🔄 **다음 단계** |
| **Phase 7** | System 1 고도화 (ACT / Diffusion Policy) | ⬜ 미착수 |

---

## Research Contributions

1. **Hierarchical Triple-System for Social HRI** — perceptual attention (System 3, CPU) + deliberative VLM reasoning (System 2, GPU, event-driven) + reactive motor execution (System 1, 50Hz)

2. **MindPower ToM + SWM** — 6-stage Theory of Mind with cross-session Social World Model persistence and situation-context generation from attention FSM

3. **Kinesthetic Teaching via HITL** — physical pose correction through direct manipulation with torque-off, encoder-capture pipeline; situation state (FSM) stored alongside each correction step

4. **Episodic LoRA for Catastrophic Forgetting Prevention** — each session distilled into a LoRA adapter; base model identity preserved; behavior accumulated over months

5. **W4A16 Quantization for Heterogeneous VLM** — monkey-patch solution for Gemma 4 Unified's interleaved sliding/full attention RoPE dimension mismatch, enabling auto_round quantization on Jetson AGX Orin (SM87)

---

## Limitations & Next Steps

| Limitation | Status | Plan |
|------------|--------|------|
| No HITL episodes collected yet | Pipeline fully ready | First HITL session (robot present) |
| ARM joints at neutral (base model) | Expected | LoRA fine-tuning after HITL collection |
| SFT only | HITL-SFT ready | DPO after 50+ episodes |
| Single adapter (no routing) | Latest adapter only | S-LoRA (Phase 7+) |
| Warm cache 0.65s only when SWM empty | SWM sliding window invalidates prefix | 2–3s speech typical with SWM history |

---

## Known Issues & Troubleshooting

주요 문제와 해결법은 **[HARU_TROUBLESHOOTING_LOG.txt](HARU_TROUBLESHOOTING_LOG.txt)** 를 참고하세요.

주요 항목 요약:

| 카테고리 | 문제 | 상태 |
|----------|------|------|
| 양자화 | bitsandbytes 4-bit SM87 비호환 | ✅ auto_round로 대체 |
| 양자화 | Jetson PyTorch 2.5 `set_submodule` 버그 → 손상 모델 | ✅ monkey-patch 적용 |
| 양자화 | Gemma 4 RoPE 차원 불일치 (512 vs 256) | ✅ monkey-patch 적용 |
| llama.cpp | `<unused49>` 비전 토큰 버그 | ❌ 포기 → vLLM 컨테이너 전환 |
| vLLM 공식 | Marlin kernel shape mismatch (이종 어텐션 비호환) | ⚠️ 우회: custom FastAPI 서버 |
| gptqmodel 2.2.0 | transformers 5.12.1 API 불일치 | ❌ 제거 → tritonv2_zp 백엔드 |
| auto-gptq | Jetson PyPI 메타데이터 버전 불일치 | ❌ pip 거부 |
| TRT-LLM Docker | ffmpeg SM87 NVCC 아키텍처 오류 | ✅ build.sh 수정 |
| TRT-LLM Docker | mooncake/nixl 테스트 실패 | ✅ --skip-tests 추가 |
| ROS2 빌드 | setuptools 81.0.0 symlink-install 실패 | ✅ 수동 install 구조 |

---

## References

### 🤖 Core Language Model & Social Intelligence

[1] Google DeepMind. **"Gemma 4 Model Card & Documentation."** 2025.
[[Docs]](https://ai.google.dev/gemma/docs/core/model_card_4)

[2] Zhou, X., Liu, J., Yerukola, A., Kim, H., & Sap, M. **"Social World Models."** *arXiv:2509.00559*, 2025.
[[Paper]](https://arxiv.org/abs/2509.00559)

[3] Taniguchi, T., Hirai, Y., Suzuki, M., Murata, S., Horii, T., & Tanaka, K. **"System 0/1/2/3: Quad-Process Theory for Multitimescale Embodied Collective Cognitive Systems."** *arXiv:2503.06138*, 2025.
[[Paper]](https://arxiv.org/abs/2503.06138)

### 📉 Quantization & Model Compression

[4] Cheng, H., et al. **"Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs (AutoRound)."** *arXiv:2309.05516*, 2023.
[[Paper]](https://arxiv.org/abs/2309.05516)

[5] Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. **"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers."** *ICLR*, 2023.
[[Paper]](https://arxiv.org/abs/2210.17323)

[6] Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., & Han, S. **"AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration."** *MLSys*, 2024.
[[Code]](https://github.com/mit-han-lab/llm-awq)

[7] Wang, S., Li, C., Kang, Y., Fan, J., Ou, Z., & Yao, A. **"SliderQuant: Accurate Post-Training Quantization for LLMs."** *ICLR*, 2026.
[[Paper]](https://arxiv.org/abs/2603.25284) [[Code]](https://github.com/deep-optimization/SliderQuant)

### ⚡ Hardware & Inference Optimization

[8] Frantar, E., Castro, R., Chen, J., Hoefler, T., & Alistarh, D. **"MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models."** *IST-DASLab*, 2024.
[[Code]](https://github.com/IST-DASLab/marlin)

[9] Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J., Zhang, H., & Stoica, I. **"Efficient Memory Management for Large Language Model Serving with PagedAttention."** *SOSP*, 2023.
[[Paper]](https://arxiv.org/abs/2309.06180)

[10] Google Cloud. **"BFloat16: The secret to high performance on Cloud TPUs."** *Google Cloud Blog*, 2019.
[[Blog]](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)

[11] NVIDIA. **"CUDA Graphs."** *CUDA C++ Programming Guide*, 2024.
[[Docs]](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

### 🧠 Fine-tuning & Continual Learning

[12] Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. **"LoRA: Low-Rank Adaptation of Large Language Models."** *ICLR*, 2022.
[[Paper]](https://arxiv.org/abs/2106.09685)

[13] et al. **"DOC: Dynamic Orthogonal Continual Fine-tuning."**
[[Code]](https://github.com/meloxxxxxx/DOC)

### 🦾 Robotics & Human-in-the-Loop RL

[14] Octo Model Team. **"Octo: An Open-Source Generalist Robot Policy."** *RSS*, 2024.
[[Project]](https://octo-models.github.io)

[15] Liu, Z., Liu, G., Qu, Y., Xue, Y., Jia, B., Yang, C., Gui, W., Huang, K., & Wang, Z. **"Preference-Calibrated Human-in-the-Loop Reinforcement Learning for Robotic Manipulation."** *arXiv:2606.03949*, 2025.
[[Paper]](https://arxiv.org/abs/2606.03949)

[16] Hejna, J., & Sadigh, D. **"Few-Shot Preference Learning for Human-in-the-Loop RL."** *CoRL*, 2022.
[[Paper]](https://arxiv.org/abs/2212.03363)
