# Project HARU: Embodied Social AI Architecture
> 최종 업데이트: 2026-06-23 | **Phase 5.5 완료 + 추론 가속 구현 진행 중** — GPU 복구 + TRT-LLM / AutoRound 경로 추가

---

## 1. 프로젝트 개요

**HARU(하루)** 는 단방향의 기계적 제어를 넘어, 물리적 실체(Physical AI)와 인간-로봇 상호작용(HRI)이 하나로 융합된 **체화된 소셜 AI(Embodied Social AI)** 기반의 능동형 반려 로봇이다.

산업용 조작 로봇이 아닌, 어떤 연령대와도 위화감 없이 교감하는 소셜 동반자(Social Companion)를 목표로 하며, 수개월~수년에 걸친 상호작용을 통해 사용자 개인의 삶에 완벽히 동기화되는 **'1인 1로봇' 생태계**를 구현한다.

---

## 2. 핵심 연구 철학 (5대 원칙)

| 원칙 | 설명 |
|------|------|
| **Pre-trained Foundation** | 물리 법칙·사회 규범 등 확고한 세계관을 지닌 대형 VLM을 베이스로 탑재, 상식 추론 능력으로 출발 |
| **SWM & Adaptive ToM** | 음성 톤·억양, 사용자에게 의미 있는 사물, 환경 맥락을 Social World Model + 적응형 마음 이론으로 통합 분석 |
| **Proactive HRI** | 명령을 기다리지 않고 로봇이 먼저 "이러한 상황 때문에 슬프신 건가요?"라고 선제적으로 다가가는 HRI |
| **Human-in-the-Loop** | 추론이 틀렸을 때 사용자의 직접 피드백(언어적·비언어적)을 즉각 수용하여 데이터화 |
| **Episodic LoRA** | 파국적 망각 방지를 위해 매일의 상호작용을 에피소드 단위 LoRA 어댑터로 모듈화·축적 |

---

## 3. 하드웨어 스펙

| 항목 | 사양 |
|------|------|
| **메인 컴퓨팅** | NVIDIA Jetson AGX Orin 64GB (ARM64) |
| **OS / 미들웨어** | Ubuntu 22.04 / ROS2 Humble |
| **AI 프레임워크** | PyTorch 2.5.0a0+nv24.08 (Jetson 전용), HuggingFace Transformers 5.12.1, PEFT (LoRA) |
| **모터 제어** | Dynamixel Protocol 2.0, U2D2 (/dev/ttyACM0, 57600 baud) |
| **카메라** | RealSense SR300 (얼굴/시선, /dev/video0) + Logitech C270 (몸통, /dev/video4) |
| **마이크** | Logitech C270 USB Audio (PulseAudio, 48kHz → 16kHz 리샘플링) |

---

## 4. HARU 모터 스펙 (Dynamixel)

### 위치 제어 관절 (Operating Mode 3)
| 관절명 | ID | 범위 (min~max) | 중립값 |
|--------|----|----------------|--------|
| r_arm_pitch | 3 | 1024 ~ 2451 | 1738 |
| l_arm_pitch | 4 | 37 ~ 1542 | 790 |
| r_shoulder_roll | 5 | 1000 ~ 2050 | 1525 |
| r_elbow_pitch | 6 | 2047 ~ 3062 | 2555 |
| l_shoulder_roll | 7 | 1047 ~ 2056 | 1552 |
| l_elbow_pitch | 8 | 1021 ~ 2007 | 1514 |
| head_pan | 10 | 1043 ~ 3071 | 2057 |
| head_tilt | 11 | 1500 ~ 3086 | 2048 |
| head_roll | 12 | 1630 ~ 2452 | 2041 |

### 속도 제어 바퀴 (Operating Mode 1)
| 관절명 | ID | 범위 | 역할 |
|--------|----|------|------|
| right_wheel | 1 | -300 ~ 300 | 오른쪽 바퀴 속도 |
| left_wheel | 2 | -300 ~ 300 | 왼쪽 바퀴 속도 |

### 표정 ID
| ID | 표정 | ID | 표정 |
|----|------|----|------|
| 0 | neutral (중립) | 4 | surprise (놀람) |
| 1 | joy (기쁨) | 5 | empathy (공감) |
| 2 | sadness (슬픔) | 6 | thinking (생각) |
| 3 | curiosity (궁금함) | 7 | concern (걱정) |

---

## 5. 핵심 아키텍처: 계층적 삼중 시스템 (Hierarchical Triple-System)

> **Phase 5.5 완료 (2026-06-22)**: Dual-System → Triple-System 진화 완성

### 왜 Triple-System인가?

**Dual-System의 한계 (해결됨):**
- 60초 고정 타이머 = 개발자가 설정한 외부 제약 → 진정한 자율이 아님
- 빈 방에서도 60초마다 Gemma 4 추론 낭비
- 사람이 등장해도 최대 60초 후 반응 → Proactive HRI 불가

**Triple-System 해결책:**
- System 3이 "지각·주의"를 전담 → 항상 켜져 있음, CPU만 사용
- Gemma 4는 "사회적으로 의미 있는 상황"에서만 깨어남
- 인식(항상) ≠ 판단(이벤트 시) → 진정한 자율 로봇

---

### System 3 — 지각·주의 (haru_attention, 경량, 항상 실행)

**담당**: 연속 지각, 사회적 상황 분류, Gemma 4 트리거 결정

**구현**: OpenCV Haar Cascade 얼굴 감지 (~5ms/frame, CPU) + 프레임 차분 모션 감지 + VAD 구독

**상황 상태 머신 (5-State FSM):**

```
[EMPTY] ─얼굴 등장─► [APPEARED] ─5s 침묵─► [PRESENT_SILENT] ─10분─► [LONG_IDLE]
   ▲                     │   ▲                    │   ▲                    │
   │                  VAD발화  │                 VAD발화 │                 VAD발화
   │                     ▼   │                    ▼   │                    ▼
   └─얼굴소멸──────── [CONVERSING] ◄─────────────────────────────────────────
                          │
                       30s 침묵
                          ▼
                   [PRESENT_SILENT]
```

| 상태 | 의미 | brain 트리거 |
|------|------|-------------|
| `EMPTY` | 방에 아무도 없음 | ❌ 없음 (낭비 방지) |
| `APPEARED` | 사람 등장, 5초 대기 중 | ❌ 대기 (먼저 말 걸 기회) |
| `APPEARED→PRESENT_SILENT` | 5초 지나도 침묵 | ✅ 즉시 트리거 |
| `CONVERSING` | 대화 중 | ✅ VAD 완료 시 즉시 |
| `PRESENT_SILENT` | 있는데 안 말함 | ✅ 120s 주기 |
| `LONG_IDLE` | 10분 이상 침묵 | ✅ 300s 주기 |

**VAD 처리 설계 (Phase 5.5 버그 수정 반영):**
- `on_vad_complete()`는 `_update()`를 경유하지 않고 직접 이벤트 생성
- VAD 감지 시 `_face_last_seen = now` 갱신 → 목소리가 들리면 사람이 있다고 간주
- 이유: 기존에 off-camera 발화 시 CONVERSING 전환 직후 face_present=False로 EMPTY 되돌아가는 버그 수정

**발행 토픽**: `/haru_attention/event` (String JSON)
```json
{
  "trigger": true,
  "state": "APPEARED_TO_SILENT",
  "context": "사람이 7초 전에 등장했고 아직 말을 걸지 않음. 먼저 인사할지 판단하세요.",
  "fsm_state": "PRESENT_SILENT",
  "face_present": true,
  "face_duration_sec": 7.2,
  "last_speech_ago_sec": null
}
```

**정책 설계 원칙:**
- attention_node = "언제 행동할 것인가" (타이밍 정책)
- Gemma 4 = "어떻게 행동할 것인가" (내용 정책, 사전학습 사회 지식 활용)
- LoRA 어댑터 = "사용자별 맞춤" (개인화 정책)

---

### System 2 — 숙고적 추론 및 소셜 상호작용 (haru_brain, Gemma 4 12B)

**담당**: 언어 이해·생성, ToM 추론, 감정 분석, 행동 계획

**선정 모델: Google Gemma 4 12B Unified** (bf16 기본 ~22GB, auto_round W4A16 양자화 ~6GB 지원)

**트리거 구조 (Phase 5.5 완료):**
- `/haru_attention/event` 구독 → 상황 컨텍스트를 user_context로 Gemma 4에 전달
- 60초 타이머 완전 제거
- 추론 중 추가 트리거 무시 (`_infer_lock`)

**침묵 선택 허용:**
- `speech: ""` 빈 문자열 → 발화 없음, 몸짓·표정만 실행
- tom_prompt.py에 침묵 규칙 명시 (상황별 예시 포함)

**System 2 출력 형식** (`haru_vla_raw` JSON):
```json
{
  "speech":            "많이 힘드셨나요?",
  "emotion":           "empathy",
  "expression_id":     5,
  "action":            {"head_tilt": 2400, "head_pan": 2057, ...},
  "duration":          2.5,
  "attention_context": "사용자가 방금 말을 마쳤음. 자연스럽게 응답하세요.",
  "attention_source":  "CONVERSING"
}
```
> `attention_context`, `attention_source`: Phase 5.5에서 추가. HITL 화면 표시 + 에피소드 데이터 저장에 사용

---

### System 1 — 반사적 행동 및 고주파 제어 (haru_action, 50Hz)

**담당**: System 2의 행동 계획을 받아 50Hz 고주파 모터 제어 실행

**구현**:
- Smoothstep 보간 (`ease = p²(3-2p)`) → 자연스러운 가감속
- 위치 제어 관절 9개 + 속도 제어 바퀴 2개 동시 처리
- 키네스테틱 모드: `_kinesthetic=True` 시 모터 쓰기 중단, 인코더 읽기만

**라우팅**:
- 직결 모드 (`hitl_mode=False`): `/haru_vla_raw` 직접 구독
- HITL 모드 (`hitl_mode=True`): `/haru_system1_command` 구독 (hitl_node 경유)

---

### 고차원 인지 프레임워크

**MindPower (Robot-Centric ToM, 6단계):**
인식 → 믿음 → 욕구 → 의도 → 결정 → 행동

Phase 5.5에서 attention_node가 "인식" 단계를 전담하여 Gemma 4에 정제된 상황 정보 전달.

**Social World Model (SWM):**
- 세션 간 장기 기억: `data/memory/swm_history.json` (50쌍 보존, 4쌍 추론 윈도우)
- 행동 패턴 기반 ToM: "사람이 들어와 말 안 거는 행동" → "탐색 중 or 수줍음" 모델링

---

## 6. ROS2 노드 구조 (Phase 5.5 Triple-System, 완성)

```
[RealSense + C270] → haru_vision → /haru_vision/compressed ─────────────────────┐
                                                                                  │
[C270 마이크]      → haru_audio  → /haru_audio/raw ───────────────────────────────┼→ haru_brain
                               → /haru_audio/vad ─────────────────────┐           │  (System 2)
                                                                       │           │
                                        haru_attention (System 3)  ←──┘           │
                                        · 얼굴 감지 (Haar Cascade)    ←────────────┘
                                        · 모션 감지 (프레임 차분)
                                        · 5-State FSM
                                                │
                                        /haru_attention/event (JSON)
                                                │
                                                ▼
                                        haru_brain (System 2, Gemma 4 12B)
                                        · MindPower ToM 6단계
                                        · SWM 세션 기억
                                        · speech="" 침묵 선택
                                                │
                        ┌───────────────────────┼───────────────────────┐
                        ▼                       ▼                       ▼
               /haru_vla_raw           /haru_expression           /haru_speech
                        │              (Int32 표정 ID)           (String 발화)
              ┌─────────┴──────────┐        │                       │
         [직결 모드]          [HITL 모드]   [디스플레이]             [TTS]
              │                    │         노드 미구현             노드 미구현
              ▼                    ▼         (Phase 6)              (Phase 6)
       haru_action          hitl_node
       (System 1)           · 인간 검토/교정
       50Hz Smoothstep      · /haru_system1_command
       [Dynamixel 11축]           │
                                  ▼
                            haru_action
                            (System 1)
```

### 전체 토픽 테이블

| 토픽 | 타입 | 발행자 | 구독자 |
|------|------|--------|--------|
| `haru_vision/compressed` | CompressedImage | haru_vision | attention, brain |
| `haru_audio/raw` | Float32MultiArray (16kHz) | haru_audio | brain |
| `haru_audio/vad` | Bool | haru_audio | attention |
| `haru_attention/event` | String JSON | attention | brain |
| `haru_vla_raw` | String JSON | brain | action(직결) / hitl_node(HITL) |
| `haru_system1_command` | String JSON | hitl_node | action(HITL 모드) |
| `haru_expression` | Int32 | brain, hitl_node, action | [디스플레이 노드 예정] |
| `haru_speech` | String | brain | [TTS 노드 예정] |
| `haru_joints/state` | Float32MultiArray 9-dim 10Hz | action | hitl_node |
| `haru_joints/torque` | Bool | hitl_node | action |
| `haru_command` | String JSON | (수동 테스트) | action |

---

## 7. 에피소드 파인튜닝 전략 (온디바이스 QLoRA)

### 추론 속도 3단계 경로 (brain_node.py 자동 선택)
| 경로 | 모델 크기 | 예상 추론 속도 | 조건 |
|------|-----------|---------------|------|
| **TRT-LLM W4A16** | ~6GB | ~6-10s | Docker 빌드 완료 + 서버 실행 후 |
| **auto_round W4A16** | ~6GB | ~10-20s | `quantize_gemma4_autoround.py` 완료 후 |
| **HF bf16 GPU** | ~22GB | ~15-30s | 즉시 사용 가능 (GPU 복구됨) |

> ✅ **GPU 복구 (2026-06-23)**: torch 2.12.1+cu130 (CUDA 13.0, 시스템 불일치) → torch 2.5.0a0+nv24.08 (CUDA 12.6 호환) 교체. 61.4 GB 통합 메모리 정상 활성화.
> ⚠️ **bitsandbytes 4-bit 불가**: SM87 CUDA 커널 미지원. auto_round/TRT-LLM 경로로 4-bit 구현.

### 메모리 계획 (Jetson 64GB)
| 항목 | 점유량 |
|------|--------|
| OS + ROS2 + 백그라운드 | ~10GB |
| Gemma 4 12B bf16 (기본) | **~22GB** |
| Gemma 4 12B auto_round W4A16 (양자화 완료 후) | **~6GB** |
| attention_node (OpenCV, CPU) | ~0.1GB |
| 파인튜닝 그래디언트 + 옵티마이저 | ~15GB |
| **가용 여유 (bf16 기준)** | **~17GB** |

### 에피소드 데이터 구조 (Phase 5.5 업데이트)
```
data/episodes/episode_YYYYMMDD_HHMMSS/
  metadata.json
  step_0000.npz  ← 각 스텝
```

`.npz` 키:
| 키 | 설명 |
|----|------|
| `image` | (448, 896, 3) uint8 — 카메라 프레임 |
| `action` | 12-dim float32 [-1,1] 정규화 (최종/교정값) |
| `action_vla` | 12-dim float32 [-1,1] 정규화 (brain 원본 예측) |
| `is_corrected` | bool |
| `speech_text` | bytes |
| `emotion` | bytes |
| `attention_source` | bytes — FSM 상태 (CONVERSING / APPEARED_TO_SILENT 등) ★신규 |
| `attention_context` | bytes — 상황 컨텍스트 문자열 ★신규 |

### QLoRA 하이퍼파라미터 (scripts/train_lora.py)
| 파라미터 | 값 |
|----------|-----|
| `load_in_4bit` | **False (bf16 전용)** |
| `r` (LoRA Rank) | 16 |
| `lora_alpha` | 32 |
| `batch_size` | 1 |
| `gradient_accumulation_steps` | 4 |
| `learning_rate` | 2e-4 |
| `target_modules` | q/k/v/o_proj + gate/up/down_proj |

> **Gemma 4 gradient_checkpointing**: `use_reentrant=False` 필수 (KV 공유 구조 버그 우회)

---

## 8. 단계별 개발 로드맵

| Phase | 목표 | 상태 |
|-------|------|------|
| **Phase 1** | Jetson 환경 구축 + 기본 추론 검증 | ✅ 완료 |
| **Phase 2** | ROS2 멀티노드 통합 (카메라→VLA→모터) | ✅ 완료 |
| **Phase 3** | HITL 에피소드 데이터 로깅 파이프라인 | ✅ 완료 |
| **Phase 4** | System 2 교체: Gemma 4 12B + MindPower ToM + SWM | ✅ 완료 2026-06-18 |
| **Phase 4.5** | 오디오 입력 노드 (haru_audio) + VAD | ✅ 완료 2026-06-18 |
| **Phase 5** | QLoRA 파이프라인 + 세션 장기 기억 + 키네스테틱 티칭 | ✅ 완료 2026-06-19 |
| **Phase 5.5** | **Triple-System**: haru_attention + 이벤트 드리븐 brain + 버그 수정 | ✅ **완료 2026-06-22** |
| **Phase 5.6** | GPU 복구 + 추론 가속 (auto_round W4A16 + TRT-LLM 경로 구현) | 🔄 **진행 중 2026-06-23** |
| **Phase 6** | Piper TTS 노드 + 표정 디스플레이 노드 | ⬜ 다음 단계 |
| **Phase 7** | System 1 고도화 (ACT / Diffusion Policy) | ⬜ 미착수 |

---

## 9. 개발 지침 & 주의사항

### 절대 원칙
- Jetson torch: 반드시 NVIDIA 전용 wheel (PyPI `pip install torch` 금지)
- 중국 모델 금지: Qwen, DeepSeek, GLM, InternVLA, Cosmos Reason2 포함
- System 3 / System 2 / System 1 분리 유지
- attention_node: CPU만 사용 (GPU는 Gemma 4 전용)
- Gemma 4: 4-bit 불가, bf16 22GB만

### setuptools 81.0.0 이슈 (빌드 주의)
- `colcon build --symlink-install`이 haru_brain / haru_attention에서 실패
- 해결: 수동 install 구조 (2026-06-22 생성 완료)
  - `install/{pkg}/lib/python3.10/site-packages/{pkg}` → symlink to src
  - `install/{pkg}/lib/python3.10/site-packages/{pkg}-{ver}-py3.10.egg-info/` (수동 생성)
- **소스 수정 시 재빌드 불필요** (symlink이므로 즉시 반영)
- 다른 패키지(haru_audio 등) 수정 시: `colcon build --symlink-install --packages-select <pkg>`

### TensorRT-LLM 빌드 진행 중 (2026-06-23)
- TRT-LLM 1.2.1: `Gemma4ForConditionalGeneration` 공식 지원 확인 (jetson-containers)
- jetson-containers 의존성: ffmpeg:8.1로 업데이트 완료 (git pull 2026-06-23)
- Docker 빌드: `scripts/build_trtllm_docker.sh` 실행 중 (백그라운드, ~8-16시간)
- 완료 후: `scripts/convert_gemma4_trtllm.sh` → `scripts/run_trtllm_server.sh` 순서
- brain_node.py가 포트 8000 서버 자동 감지 → 최속 경로 선택

### 현재 워크스페이스 구조
```
robot_brain_workspace/
├── haru_vla_env/              ← Python venv (torch 2.5.0a0+nv24.08, transformers 5.12.1 등)
├── scripts/
│   ├── test_gemma4.py
│   ├── train_lora.py              ← QLoRA 파인튜닝
│   ├── convert_to_rlds.py
│   ├── quantize_gemma4_autoround.py  ← W4A16 양자화 실행 스크립트 ★ 신규
│   ├── build_trtllm_docker.sh        ← TRT-LLM Docker 빌드 ★ 신규
│   ├── convert_gemma4_trtllm.sh      ← TRT-LLM 엔진 변환 ★ 신규
│   └── run_trtllm_server.sh          ← TRT-LLM 서버 실행 ★ 신규
├── data/
│   ├── episodes/              ← HITL 수집 데이터 (attention_source/context 포함)
│   ├── adapters/              ← 학습된 LoRA 어댑터
│   ├── memory/                ← SWM 이력 (swm_history.json)
│   ├── gemma4_autoround_w4a16/ ← auto_round W4A16 양자화 모델 (양자화 완료 후 생성)
│   ├── trtllm/engine/         ← TRT-LLM 엔진 (빌드 완료 후 생성)
│   └── trtllm/visual_engine/  ← TRT-LLM 비전 인코더
├── src/
│   ├── haru_vision/           ← 카메라 노드 (듀얼, 3Hz, 896×448 JPEG)
│   ├── haru_audio/            ← 마이크 노드 (VAD, 16kHz, C270)
│   ├── haru_attention/        ← System 3: 주의 노드 ★ Phase 5.5
│   │   └── attention_node.py  ← 얼굴감지 + 5-State FSM + VAD통합
│   ├── haru_brain/            ← System 2 (Gemma 4 12B)
│   │   ├── brain_node.py          ← 3-tier 자동 선택 (TRT-LLM→AutoRound→HF)
│   │   ├── gemma4_inference.py    ← HF bf16 기본 경로
│   │   ├── gemma4_trtllm_inference.py   ← TRT-LLM 고속 경로 ★ 신규
│   │   ├── gemma4_autoround_inference.py ← auto_round 중속 경로 ★ 신규
│   │   ├── tom_prompt.py          ← 침묵 규칙 + 상황 컨텍스트
│   │   ├── session_memory.py
│   │   └── adapter_manager.py
│   ├── haru_action/           ← System 1 (50Hz Smoothstep)
│   └── haru_logger/           ← HITL 로거 (attention_source/context 저장)
├── torch-2.5.0a0+...nv24.08...whl  ← Jetson 전용 torch wheel (사용됨)
├── launch_vla.sh              ← 실행 스크립트
├── HARU_PROJECT_CONTEXT.md    ← 이 파일
├── HARU_RESEARCH_GOALS.txt
└── HARU_RUN_COMMANDS.txt
```
