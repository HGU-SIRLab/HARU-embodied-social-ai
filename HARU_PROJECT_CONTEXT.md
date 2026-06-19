# Project HARU: Embodied Social AI Architecture
> 최종 업데이트: 2026-06-19 | Phase 5 코드 완성 + 전체 버그수정 완료 (키네스테틱 티칭 + QLoRA + 세션 장기 기억)

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
| **AI 프레임워크** | PyTorch, HuggingFace Transformers, PEFT (LoRA) |
| **모터 제어** | Dynamixel Protocol 2.0, U2D2 (/dev/ttyACM0, 57600 baud) |
| **카메라** | RealSense SR300 (얼굴/시선, /dev/video0) + Logitech C270 (몸통, /dev/video4) |
| **마이크** | Logitech C270 USB Audio (PulseAudio, 48kHz → 16kHz 리샘플링, Phase 4.5 구현완료) |

---

## 4. HARU 모터 스펙 (Dynamixel)

### 위치 제어 관절 (Operating Mode 3)
| 관절명 | ID | 범위 (min~max) | 역할 |
|--------|----|----------------|------|
| r_arm_pitch | 3 | 1024 ~ 2451 | 오른팔 피치 |
| l_arm_pitch | 4 | 37 ~ 1542 | 왼팔 피치 |
| r_shoulder_roll | 5 | 1000 ~ 2050 | 오른쪽 어깨 롤 |
| r_elbow_pitch | 6 | 2047 ~ 3062 | 오른쪽 팔꿈치 |
| l_shoulder_roll | 7 | 1047 ~ 2056 | 왼쪽 어깨 롤 |
| l_elbow_pitch | 8 | 1021 ~ 2007 | 왼쪽 팔꿈치 |
| head_pan | 10 | 1043 ~ 3071 | 고개 좌우 (도리도리) |
| head_tilt | 11 | 1500 ~ 3086 | 고개 앞뒤 (끄덕임) |
| head_roll | 12 | 1630 ~ 2452 | 고개 기울기 |

### 속도 제어 바퀴 (Operating Mode 1)
| 관절명 | ID | 범위 | 역할 |
|--------|----|------|------|
| right_wheel | 1 | -300 ~ 300 | 오른쪽 바퀴 속도 |
| left_wheel | 2 | -300 ~ 300 | 왼쪽 바퀴 속도 |

### 표정 ID (디스플레이)
| ID | 표정 |
|----|------|
| 0 | neutral (중립) |
| 1 | joy (기쁨) |
| 2 | sadness (슬픔) |
| 3 | curiosity (궁금함) |
| 4 | surprise (놀람) |
| 5 | empathy (공감) |
| 6 | thinking (생각) |
| 7 | concern (걱정) |

---

## 5. 핵심 아키텍처: 계층적 이중 시스템 (Hierarchical Dual-System)

> OpenVLA 단일 모델의 한계(언어 능력 붕괴, 15초 지연)를 극복하기 위해
> 다니엘 카너먼의 System 1 / System 2 인지 이론을 로봇 아키텍처에 적용

### OpenVLA 폐기 이유 (기록)
- **이산적 행동 토큰화**: LLM의 의미론적 공간을 물리 좌표계로 강제 편향 → 언어 생성 능력 붕괴 (ŸŸŸŸ 현상)
- **추론 지연 15초**: 자기회귀 방식으로 행동 토큰 생성 → 실시간 HRI 불가
- **단일 모델 병목**: 대화와 제스처를 동시에 수행하는 멀티태스킹 구조적 불가능

---

### System 2 — 숙고적 추론 및 소셜 상호작용 (High-Level VLM)

**담당**: 언어 이해·생성, ToM 추론, 감정 분석, 행동 계획

**선정 모델: Google Gemma 4 12B Unified** (2026.06 출시)

| 특징 | 설명 |
|------|------|
| **인코더-프리 아키텍처** | 별도 비전·오디오 인코더 없이 원시 픽셀·오디오를 단일 디코더에 직접 주입 → 엣지 메모리 최적화 |
| **네이티브 오디오 처리** | 16kHz 원시 파형 → 40ms 단위 640샘플 → 선형 투영 → 3,840차원 임베딩. 감정·억양 정보 보존 |
| **다중 토큰 예측 (MTP)** | 투기적 디코딩 기반 병렬 토큰 생성 → 대화 지연 대폭 감소 |
| **Jetson 적합성** | bf16 로드 시 **~22GB** 점유 (bitsandbytes 4-bit 비호환 확인됨), Orin 64GB에서 운용 가능 |

**차선책: Microsoft Phi-4-Multimodal-Instruct** (2025.02, 5.6B)
- Mixture-of-LoRAs 구조 (비전 LoRA + 오디오 LoRA 분리)
- WhisperV3 대비 우수한 ASR (WER 6.14%)
- 메모리 제약이 심할 경우 우선 적용

**System 2 출력 형식** (JSON → `haru_vla_raw` 토픽 → action_node 직결 또는 HITL 경유):
```json
{
  "speech": "많이 힘드셨나요?",
  "emotion": "empathy",
  "expression_id": 5,
  "action": {
    "head_tilt": 2400, "head_pan": 2057, "head_roll": 2041,
    "r_arm_pitch": 1738, "l_arm_pitch": 790,
    "r_shoulder_roll": 1525, "r_elbow_pitch": 2555,
    "l_shoulder_roll": 1552, "l_elbow_pitch": 1514,
    "right_wheel": 0.0, "left_wheel": 0.0
  },
  "duration": 2.5
}
```

---

### System 1 — 반사적 행동 및 고주파 제어 (Low-Level Policy)

**담당**: System 2의 행동 계획을 받아 50Hz 고주파 모터 제어 실행

**현재 구현**: `haru_action` 노드 (Smoothstep 보간, 50Hz)
- System 2 명령 수신 → 관절 9개 위치 보간 + 바퀴 2개 속도 제어
- System 2가 다음 응답을 생성하는 동안에도 독립적으로 부드러운 움직임 유지

**향후 고도화** (Phase 5+): ACT 또는 Diffusion Policy 기반 경량 정책 모델로 교체

---

### 고차원 인지 프레임워크

**MindPower (Robot-Centric ToM)**
- 인식(Perception) → 믿음(Belief) → 욕구(Desire) → 의도(Intention) → 결정(Decision) → 행동(Action)
- 1인칭 로봇 관점에서 사용자의 잘못된 믿음 수정, 묵시적 목표 유추

**Social World Model (SWM)**
- S3AP 구조화 공식으로 사용자의 숨겨진 의도·정신 상태 변화 명시적 추적
- 과거 에피소드가 미래 상호작용에 미칠 파급 효과 예측 → 공감 대화 스크립트 생성

---

## 6. ROS2 노드 구조

```
[마이크] ──────────────────────────────────────────────────────────┐
                                                                   ↓
[RealSense]─┐                                          ┌─ haru_audio ─┐
[C270]      ├→ haru_vision → /haru_vision/compressed → │              │
            ┘                                          │  haru_brain  │→ /haru_speech → [TTS]
                                                       │  (System 2)  │
                                                       │ Gemma 4 12B  │→ /haru_expression → [디스플레이]
                                                       └──────────────┘
                                                              ↓
                                                   /haru_system1_command
                                                              ↓
                                                   haru_action (System 1)
                                                    50Hz Smoothstep 보간
                                                              ↓
                                                    [Dynamixel U2D2]
                                                    관절 9개 + 바퀴 2개

[HITL 모드] haru_logger ─ 에피소드 수집 → data/episodes/
```

| 토픽 | 타입 | 방향 |
|------|------|------|
| `haru_vision/compressed` | CompressedImage | vision → brain |
| `haru_audio/raw` | Float32MultiArray (16kHz float32 PCM) | audio → brain |
| `haru_audio/vad` | Bool | audio → (모니터링) |
| `haru_vla_raw` | String (JSON) | **brain → action (직결)** 또는 brain → hitl_node |
| `haru_system1_command` | String (JSON) | hitl_node → action **(HITL 모드 전용)** |
| `haru_expression` | Int32 | brain/hitl → 디스플레이 |
| `haru_speech` | String | brain → TTS |
| `haru_joints/state` | Float32MultiArray (9-dim, 10Hz) | action → hitl_node (키네스테틱 티칭) |
| `haru_joints/torque` | Bool | hitl_node → action (True=ON, False=OFF) |

---

## 7. 에피소드 파인튜닝 전략 (온디바이스 QLoRA)

### 메모리 계획 (Jetson 64GB 통합 메모리)
| 항목 | 점유량 |
|------|--------|
| OS + ROS2 + 백그라운드 | ~10GB |
| Gemma 4 12B **(bf16, 4-bit 비호환)** | **~22GB** |
| 파인튜닝 그래디언트 + 옵티마이저 | ~15GB |
| **가용 여유** | **~17GB** |

> ⚠️ **4-bit 불가 주의**: bitsandbytes 0.49.2 + Gemma 4 Unified 아키텍처 조합에서 `'model' is not an nn.Module` 오류 발생 확인 (2026-06-18). **bf16 22GB가 유일하게 동작하는 구성.** `train_lora.py`도 동일하게 bf16 로드 (`load_in_4bit=False`).

### PEFT + 수동 학습루프 QLoRA 하이퍼파라미터 (scripts/train_lora.py)
| 파라미터 | 값 | 근거 |
|----------|-----|------|
| `load_in_4bit` | **False (bf16 전용)** | bitsandbytes 4-bit + Gemma 4 Unified 비호환 확인 (2026-06-18) |
| `r` (LoRA Rank) | 16 | 표현력·효율 균형 |
| `lora_alpha` | 32 | Rank × 2, 학습 안정성 |
| `batch_size` | 1 | VRAM 피크 억제 |
| `gradient_accumulation_steps` | 4 | 실질 배치 효과 |
| `learning_rate` | 2e-4 | 기본값 (진동 시 1e-4로 낮추기) |
| `target_modules` | q/k/v/o_proj + gate/up/down_proj | Attention + MLP 전체 학습 |

> **Gemma 4 gradient_checkpointing 주의**: 후반 18개 레이어에서 KV 캐시를 공유하는 구조(`num_kv_shared_layers`)로 인해 기본 `use_reentrant=True` 설정 시 어텐션 붕괴 버그 발생. `train_lora.py`에서 `gradient_checkpointing_kwargs={'use_reentrant': False}`로 우회 적용 완료.

### S-LoRA 동적 어댑터 서빙 (에피소드 기억)
- 백본 모델(Gemma 4 12B)은 메모리에 1회만 로드
- 에피소드 유형별 LoRA 어댑터 (수십 MB 단위) 디스크↔메모리 동적 스왑
- 맥락 인식형 라우팅: 첫 3초 오디오 텐서 분석 → 사용자/상황 식별 → 해당 어댑터 로드
- 파국적 망각 원천 차단: 새 에피소드는 새 어댑터에 학습, 기존 지식 보존

---

## 8. 단계별 개발 로드맵

| Phase | 목표 | 상태 |
|-------|------|------|
| **Phase 1** | Jetson 환경 구축 + OpenVLA 단독 추론 검증 | ✅ 완료 |
| **Phase 2** | ROS2 멀티노드 통합 (카메라→VLA→모터) | ✅ 완료 (OpenVLA로) |
| **Phase 3** | HITL 에피소드 데이터 로깅 파이프라인 | ✅ 완료 |
| **Phase 4** | **System 2 교체**: Gemma 4 12B Unified 도입 + MindPower ToM + SWM | ✅ **완료** (2026-06-18) |
| **Phase 4.5** | 오디오 입력 노드 (`haru_audio`) + C270 마이크 + Gemma 4 네이티브 오디오 | ✅ **완료** (2026-06-18) |
| **Phase 5** | QLoRA 에피소드 파인튜닝 + LoRA 어댑터 서빙 + 세션 간 장기 기억 + 키네스테틱 티칭 | ✅ **코드 완성** (2026-06-19, 실제 에피소드 수집 필요) |
| **Phase 6** | System 1 고도화 (ACT / Diffusion Policy) | ⬜ 미착수 |

---

## 9. AI 엔지니어를 위한 개발 지침

### 절대 원칙
- 헷갈릴 때는 항상 이 파일을 먼저 확인
- Jetson torch는 반드시 NVIDIA 전용 wheel 사용 (PyPI torch 설치 금지)
- 중국 모델 사용 금지 (Qwen, DeepSeek, GLM, InternVLA, Cosmos Reason2 포함)
- System 2와 System 1은 반드시 분리 유지 (단일 엔드투엔드 모델로 합치지 말 것)

### TensorRT-LLM 적용 불가 판정 (2026-06-18 검토)
- **판정: 현재 불가** — jetson-containers 이미지 최신 버전(2024-11-13 기준) 기준 Gemma 4 `gemma4_unified` 아키텍처 미지원
- aarch64 TRT-LLM wheel 없음; `gemma4_unified` 전용 변환 스크립트 없음
- **대안**: Gemma 3용 TRT-LLM 프로파일 존재하나 Gemma 4 Unified와 구조 불일치 (인코더-프리 MTP 구조)
- **재검토 시점**: jetson-containers Gemma 4 지원 이미지 출시 후 (미정)

### 현재 워크스페이스 구조
```
robot_brain_workspace/
├── haru_vla_env/          ← Python venv (transformers 5.12, bitsandbytes, torch NV, peft, trl)
├── scripts/
│   ├── test_gemma4.py     ← Gemma 4 추론 단독 테스트
│   ├── train_lora.py      ← Phase 5 QLoRA 파인튜닝 (HITL 에피소드 → LoRA 어댑터)
│   └── convert_to_rlds.py ← RLDS 변환
├── data/
│   ├── episodes/          ← HITL 수집 데이터 (step_XXXX.npz)
│   ├── adapters/          ← 학습된 LoRA 어댑터 (adapter_YYYYMMDD_HHMMSS/)
│   └── memory/            ← 세션 간 SWM 이력 (swm_history.json)
├── src/
│   ├── haru_vision/       ← 카메라 노드 (듀얼 카메라, 3Hz, SSH 폴백)
│   ├── haru_brain/        ← System 2 (Gemma 4 12B, MindPower ToM, SWM, LoRA 자동 로드)
│   │   ├── brain_node.py
│   │   ├── gemma4_inference.py  ← PEFT 어댑터 자동 로드 포함
│   │   ├── tom_prompt.py        ← 세션 간 장기 기억 영속화 포함
│   │   ├── session_memory.py    ← SWM 디스크 저장/복원 (Phase 5 신규)
│   │   └── adapter_manager.py   ← LoRA 어댑터 관리 (Phase 5 신규)
│   ├── haru_audio/        ← 오디오 노드 (C270 PulseAudio, VAD, 48→16kHz 리샘플링)
│   ├── haru_action/       ← System 1 (50Hz Smoothstep, hitl_mode 파라미터)
│   └── haru_logger/       ← HITL 로거 (12-DoF 저장)
├── launch_vla.sh          ← 실행 스크립트 (all/audio/hitl/brain/action/vision/audio_only)
└── HARU_RUN_COMMANDS.txt  ← 명령어 모음
```

### Phase 5 완성 내용 (2026-06-19, 코드 완성 — 실제 에피소드 수집 필요)

**세션 간 장기 기억 (SWM 영속 레이어)**
- `session_memory.py`: `data/memory/swm_history.json`에 최대 50쌍 보존
- 추론 윈도우: 최근 4쌍만 메시지에 포함 (토큰 절약)
- `tom_prompt.py`: 시작 시 자동 로드, 대화마다 자동 저장 (원자적 tmp+rename 쓰기)

**키네스테틱 티칭 (물리적 교정)**
- HITL [C] → [D] 선택 시: `haru_joints/torque=False` publish → 사용자가 손으로 조작 → Enter → 인코더 자동 캡처
- 구현: `MultiThreadedExecutor` + `ReentrantCallbackGroup`으로 `input()` 블록 중에도 `/haru_joints/state` 콜백 실행 가능
- `_kb_worker`와 `_run_correction` 간 stdin 충돌: `select()` 0.2s 타임아웃 + `_in_correction` 플래그로 해결
- `POSITION_JOINT_ORDER` 순서가 `action_node.HARU_LIMITS` 순서와 반드시 일치해야 Float32MultiArray 인덱스 매핑 정상 동작

**QLoRA 학습 파이프라인**
- `scripts/train_lora.py`: `HaruEpisodeDataset` (PyTorch Dataset, 사전 토큰화)
- 정확한 loss 마스킹: `prompt_len` 기준 `labels[:prompt_len]=-100` (프롬프트 구간 제외)
- 어댑터 저장: `data/adapters/adapter_YYYYMMDD_HHMMSS/`
- `brain_node` 재시작 시 `adapter_manager.py`가 mtime 기준 최신 어댑터 자동 선택·로드

### Phase 4 완료 시점 주요 사실 (2026-06-18)
- **Gemma 4 12B**: bf16으로만 로드 가능 (bitsandbytes 4-bit 비호환), 22GB VRAM 사용
- **추론 속도**: 약 45~50초 / 응답 (5 tokens/sec, Jetson AGX Orin 한계)
- **inference_interval 기본값**: 60초
- **JSON 출력 품질**: emotion ↔ expression_id 매핑 정확, 관절 범위 준수, 한국어 자연어 발화
- **SWM 연속성**: user-assistant 쌍 4턴 이력 유지
- **토픽 라우팅**: brain → `haru_vla_raw` → (HITL 또는 직결) → action_node
- **action_node 파라미터**: `hitl_mode:=true` (HITL), 기본=직결모드
- **HITL 데이터**: 12-DoF 정규화 저장 (표정+관절9+바퀴2)
