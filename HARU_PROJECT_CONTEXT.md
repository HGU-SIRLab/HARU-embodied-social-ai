# Project HARU: Embodied Social AI Architecture
> 최종 업데이트: 2026-07-02 | **Phase 6.7 완료** — Google 공식 QAT 체크포인트 프로덕션 교체 (PPL -21.4%, 속도 -4.3%) | 이전: Phase 6.5.5 Robot-display-HRI anime 얼굴 FULLSCREEN 통합, 14 감정, AI-only 제어, speech 0.65s warm (69×), 7/7 노드

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

### 표정 ID (Phase 6.5.5 — Robot-display-HRI 통합)
> 출처: https://github.com/HGU-SIRLab/Robot-display-HRI (pygame anime 얼굴, 14감정)
> AI(/haru_expression Int32)가 EMOTION_MAP으로 변환 → pygame emotion_queue → 렌더러

| HARU ID | 감정명 | Robot-display-HRI 키 |
|---------|--------|---------------------|
| 0 | neutral (중립) | NEUTRAL |
| 1 | joy (기쁨) | HAPPY |
| 2 | sadness (슬픔) | SAD |
| 3 | curiosity (궁금함) | THINKING |
| 4 | surprise (놀람) | SURPRISED |
| 5 | empathy (공감) | TENDER |
| 6 | thinking (생각) | THINKING |
| 7 | concern (걱정) | SCARED |

내부 전용 감정 (brain이 직접 출력하지 않음): EXCITED, ANGRY, LISTENING, CLOSE, SCANNING, SLEEPY, WAKE

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

**선정 모델: Google Gemma 4 12B Unified** (bf16 기본 ~22GB / auto_round W4A16 ~6GB ✅ / TRT-LLM W4A16 ~6GB 🔄)

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

### 추론 속도 경로 (brain_node.py 자동 선택)
| 경로 | 모델 크기 | 추론 속도 | 상태 |
|------|-----------|-----------|------|
| **vLLM 컨테이너 서버 (포트 8000)** | ~7.4GB | **현재 2.8 tok/s ≈ 64s** (목표: 20+ tok/s ≈ 5s) | ✅ **운용 중 (Phase 5.7)** |
| **auto_round W4A16 직접 로드** | ~7.4GB | ~10-20s | ✅ 완료 (2026-06-23) |
| **HF bf16 GPU** | ~22GB | ~15-30s | ✅ 항상 사용 가능 |

> ✅ **GPU 복구 (2026-06-23)**: torch 2.12.1+cu130 (CUDA 13.0 불일치) → torch 2.5.0a0+nv24.08 (CUDA 12.6 호환). 61.4 GB 통합 메모리 활성화.
> ❌ **auto_round W4A16 초기 시도 실패 (2026-06-23)**: 저장 모델이 손상됨 — qweight 없음, 349 키만 저장. 원인: Jetson PyTorch 2.5 `set_submodule` exact type check 버그 (`type(mod) is not nn.Module`). 상세 버그 체인은 아래 참조.
> ✅ **Jetson set_module 패치 + 양자화 완료 (2026-06-23)**: `_patched_set_module` monkey-patch(`getattr`/`setattr` 직접 탐색)로 버그 우회. 128-sample 프로덕션 완료 — 2 shard, 1333 키, 328 qweight, 7.3GB, 28.8분 소요.
> ⚠️ **bitsandbytes 4-bit 불가**: SM87 CUDA 커널 미지원. auto_round/TRT-LLM 경로로 4-bit 구현.

#### Gemma 4 Unified 양자화 기술 요점

**해결된 문제 1 — RoPE 차원 불일치:**
Gemma 4는 6레이어마다 `sliding_attention`(head_dim=256) / `full_attention`(global_head_dim=512)이 교대 배치됩니다. auto_round 0.13.1의 블록 단위 캘리브레이션이 position_embeddings를 재사용하면서 512 vs 256 차원 불일치 발생 → `apply_rotary_pos_emb` monkey-patch로 해결 (`scripts/quantize_gemma4_autoround.py` 참조).

**해결된 문제 2 — Jetson PyTorch 2.5 `set_submodule` 버그 (2026-06-23 발견):**

버그 체인:
1. Jetson PyTorch 2.5의 `set_submodule`: `type(mod) is not torch.nn.Module` (exact type check, isinstance 아님) → 모든 nn.Module 서브클래스에서 AttributeError 발생
2. auto_round의 `set_module()` → `try/except (AttributeError, KeyError): return` → 조용히 무시
3. `pack_layer()` → `set_module(model, layer_name, QuantLinear)` → QuantLinear 모델에 설치 안 됨 (silent no-op)
4. `release_layer_safely(layer)` → 원본 nn.Linear의 `weight = None` 설정 → weight 파괴
5. `model.state_dict()` → None weight 제외, 349개 non-linear 키만 반환
6. 저장된 safetensors: qweight 없음, 레이어 weight 없음 → 완전히 손상된 모델

증상: 저장 파일 단일 shard, 349 키 (layernorm·embedding·layer_scalar만 존재)
기대: 2 shard, 1333 키, 328 qweight

**패치 (적용됨 — `scripts/quantize_gemma4_autoround.py`):**
```python
def _patched_set_module(model, key, new_module):
    atoms = key.split(".")
    name = atoms[-1]
    parent = model
    for atom in atoms[:-1]:
        parent = getattr(parent, atom)
    setattr(parent, name, new_module)

# 세 모듈 모두 교체
import auto_round.utils.model as _ar_model_mod
_ar_model_mod.set_module = _patched_set_module
_ar_export.set_module = _patched_set_module  # export 모듈
```

검증: 진단 실행 (4 샘플) → 328 qweight, 1333 키, 2 shard 확인.

#### W4A16 PPL 검증 결과 (2026-07-01, RTN 최초 측정)

측정 방식: logprob 기반 통일 계산 (BOS 제외, vLLM `echo+logprobs`와 동등)
캘리브레이션 코퍼스: 한국어 대화·감정 15문장 + 영어 HRI 관련 15문장 (총 30문장)

| 구성 | 코퍼스 PPL | bf16 대비 | 비고 |
|------|-----------|----------|------|
| bf16 원본 | 1,443 | 기준선 | HF 직접 추론 |
| bf16 + RoPE 패치 | 1,443 | **+0.00%** ✅ | 패치 수치 영향 없음 |
| W4A16 vLLM 서버 | 2,569 | +78% | 실제 운용 경로 |

**RoPE 패치 수치 영향 +0.00%**: 30개 문장 손실값이 소수점 6자리까지 동일. 논문 서술 가능.

**W4A16 PPL +78% 원인**: `ITERS=0` (RTN 모드) — AutoRound SignRound 최적화 미적용.
vLLM은 이종 어텐션 인식 후 TRITON_ATTN 커널이 RoPE를 직접 처리하므로
Python 레벨 RoPE 패치와 무관. PPL 열화는 순수 RTN 양자화 특성.

#### Phase 6.7 — Google 공식 QAT 체크포인트로 교체 (2026-07-02)

RTN 대신 Google이 2026-06-05 공식 배포한 QAT(양자화 인식 학습) 체크포인트
(`google/gemma-4-12B-it-qat-w4a16-ct`, compressed-tensors 포맷)를 프로덕션에 적용.
RTN·QAT를 **동일 스크립트·동일 30문장 코퍼스·동일 vLLM echo+logprobs 방식**으로
나란히 재측정(apples-to-apples):

| 구성 | 코퍼스 PPL | 속도 (decode) |
|------|-----------|---------------|
| W4A16 RTN (재측정, 2026-07-02) | 3298.28 | 18.87 tok/s |
| W4A16 QAT (2026-07-02) | **2592.03 (-21.4%)** | 18.06 tok/s (-4.3%) |

> 위 표의 RTN=3298.28은 이 재측정에서 나온 값으로, 위 2026-07-01 표의 2,569와 다름
> (당시 측정 스크립트가 정확히 기록되지 않아 재현 불가 — 이번엔 RTN·QAT를 완전히
> 동일한 조건에서 나란히 측정해 상대 비교 신뢰도를 확보함). 절대 PPL이 두 모델 다
> 큰 이유: 코퍼스가 "사용자가 로봇에 할 법한 개방형 질문"이라 instruction-tuned
> 모델 입장에서 정답 분포가 넓어 원래 절대값 자체가 크다 (bf16도 1,443).

**패치**: 코드 변경 없이 config.json 2개 필드만 수정 (기존 `gemma4_mm_patch.py` 그대로 재사용)
- `architectures`: `Gemma4UnifiedForConditionalGeneration` → `Gemma4ForConditionalGeneration`
  (vLLM 0.21.0 registry.py 미등록 이름이라 패치가 이미 처리하는 이름으로 정정)
- `vision_config.default_output_length: 280` 추가 (HF 공식 config엔 `num_soft_tokens`만 있고
  vLLM 패치가 참조하는 필드명이 없어 `AttributeError` 발생 → 동일값으로 추가)

**검증**: 실제 HARU 시스템 프롬프트(MindPower 6단계 + JSON 스키마) + 이미지 입력 + 스트리밍으로
`gemma4_trtllm_inference.py`(brain_node 1순위 경로) 재현 테스트 → JSON 스키마 정상 출력,
침묵 규칙("생각에 잠긴 상황" → `speech:""`, `emotion:"thinking"`)도 정확히 재현됨.
`haru_all` 무수정으로 정상 동작 확인.

**모델 위치**: `data/gemma4_vllm_patched/` = QAT (현재 프로덕션).
RTN 백업: `data/gemma4_vllm_patched_rtn_backup/` (롤백 시 두 디렉토리명 맞바꾸면 됨).
`scripts/run_vllm_server.sh`는 무수정.

**재양자화(SignRound) 계획 폐기**: QAT 교체로 목표 달성, GROUP_SIZE=64/SignRound(ITERS=200)
재시도 계획은 더 이상 불필요.

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
| **Phase 5.6** | GPU 복구 + TRT-LLM 경로 구현 + Jetson PyTorch 버그 패치 + auto_round W4A16 양자화 | ✅ **완료 2026-06-23** |
| **Phase 5.7** | vLLM 컨테이너 기반 Gemma4 서버 + 비전(VLM) 지원 + brain_node 통합 테스트 | ✅ **완료 2026-06-25** |
| **Phase 5.8** | 추론 속도 가속 (2.8 tok/s → 20+ tok/s, ~64s → ~5s) | 🔄 **진행 예정** |
| **Phase 6.1** | Piper TTS 노드 (edge-tts ko-KR-SunHiNeural) | ✅ 완료 2026-06-26 |
| **Phase 6.2** | 표정 디스플레이 노드 (pygame 8감정, haru_all 7/7) | ✅ 완료 2026-06-26 |
| **Phase 6.3** | Streaming + prefix caching (speech 0.65s warm, 69×) | ✅ 완료 2026-06-26 |
| **Phase 6.4** | action_ready_cb + 약어 키 (7.4s full inference) | ✅ 완료 2026-06-26 |
| **Phase 6.5** | HITL 에피소드 수집 + 첫 LoRA 어댑터 (6.38s 중앙값) | ✅ 완료 |
| **Phase 6.5.5** | **Robot-display-HRI 통합** — 14 anime 감정, FULLSCREEN, AI-only 제어 | ✅ **완료 2026-07-01** |
| **Phase 6.7** | **QAT 체크포인트 교체** — Google 공식 QAT(compressed-tensors), PPL -21.4%, 속도 -4.3% | ✅ **완료 2026-07-02** |
| **Phase 7** | System 1 고도화 (ACT / Diffusion Policy) | ⬜ 미착수 |

---

## 9. 개발 지침 & 주의사항

### 절대 원칙
- Jetson torch: 반드시 NVIDIA 전용 wheel (PyPI `pip install torch` 금지)
- 중국 모델 금지: Qwen, DeepSeek, GLM, InternVLA, Cosmos Reason2 포함
- System 3 / System 2 / System 1 분리 유지
- attention_node: CPU만 사용 (GPU는 Gemma 4 전용)
- Gemma 4: 4-bit 불가, bf16 22GB만
- **Jetson PyTorch 2.5 `set_submodule` 버그**: `type(mod) is not nn.Module` 정확한 타입 체크로 모든 서브클래스에서 AttributeError → auto_round `set_module` 조용히 무시 → QuantLinear 설치 실패 → 원본 weight 파괴. `_patched_set_module`(`getattr/setattr` 직접 탐색)로 우회 적용됨.

### setuptools 81.0.0 이슈 (빌드 주의)
- `colcon build --symlink-install`이 haru_brain / haru_attention에서 실패
- 해결: 수동 install 구조 (2026-06-22 생성 완료)
  - `install/{pkg}/lib/python3.10/site-packages/{pkg}` → symlink to src
  - `install/{pkg}/lib/python3.10/site-packages/{pkg}-{ver}-py3.10.egg-info/` (수동 생성)
- **소스 수정 시 재빌드 불필요** (symlink이므로 즉시 반영)
- 다른 패키지(haru_audio 등) 수정 시: `colcon build --symlink-install --packages-select <pkg>`

### Phase 5.7~6.3 — 추론 가속 전체 경로 (완료 2026-06-26)

#### 현재 추론 아키텍처
- **서버**: vLLM 0.21.0 컨테이너, W4A16 QAT(Google 공식, compressed-tensors) + Marlin INT4 GEMM + CUDAGraph (Phase 6.7, 이전: AutoRound RTN)
- **모델**: `data/gemma4_vllm_patched/` = QAT 원본 `data/gemma4_qat_w4a16/` 패치본 (Phase 6.7) + `gemma4_mm_patch.py` (VisionEmbedder 패치, 코드 무수정 재사용)
- **속도**: 18.06 tok/s (QAT, Phase 6.7 재측정) — 이전 RTN 19.3~19.6 tok/s 대비 -4.3%
- **시작**: `bash scripts/run_vllm_server.sh`

#### 체감 지연 최적화 (Phase 6.3)
| 최적화 | 효과 |
|-------|------|
| `stream=True` + `_extract_speech_field` | speech 필드 완성 즉시 TTS 발행 (~0.65s warm) |
| `_extract_expression_id` | expression_id 완성 즉시 표정 변경 (~1.32s warm) |
| `--enable-prefix-caching` | 시스템 프롬프트 383 tok KV 캐시 (TTFT ~0.3s warm) |
| IMG_SIZE 448→336 | 비전 토큰 44% 감소, 프리필 ~0.6s 단축 |
| SWM window 4→2 + 응답 압축 | ~300 tok/턴 절약 |
| **합계** | **HF bf16 ~45s → 0.65s warm = 69×** |

#### haru_all 7/7 노드 (Phase 6.2+ 기준)
```
vision_node      (3Hz, 336×336 JPEG, 듀얼 카메라)
attention_node   (5-State FSM, OpenCV 얼굴/모션/VAD, CPU)
audio_node       (16kHz VAD, C270)
brain_node       (Gemma 4 12B, 스트리밍, prefix caching)
action_node      (50Hz Smoothstep, 11관절 전체 제어)
tts_node         (edge-tts ko-KR-SunHiNeural, 400ms, mpg123 hw:3,0)
expression_node  (Robot-display-HRI pygame FULLSCREEN 1024×600, 14감정, DISPLAY=:0)
```

#### 실제 출력 로그 예 (Phase 6.3 이후)
```
[Brain] ✅ TRT-LLM 서버 연결 성공 — 고속 추론 모드 (W4A16 INT4)
[Brain] [STREAM] speech 조기 발행: '안녕하세요! 저와 함께 시간을 보낼 준비가 되셨나요?'
[Brain] [STREAM] expression 조기 발행: 1
[Brain] [face_detected] 10.5s | emotion=joy | speech="안녕하세요! 저와..."
[Pub] expr=1 speech='안녕하세요! 저와 함께 시간을...' head=(2100,2057,2041) r_arm=1738/1525/2555 l_arm=790/1552/1514 wheel=(0,0)
```

### 현재 워크스페이스 구조
```
robot_brain_workspace/
├── haru_vla_env/              ← Python venv (torch 2.5.0a0+nv24.08, transformers 5.12.1 등)
├── scripts/
│   ├── test_gemma4.py             ← bf16 기본 추론 테스트
│   ├── test_autoround.py          ← W4A16 양자화 전체 기능 테스트 ★ 신규
│   ├── train_lora.py              ← QLoRA 파인튜닝
│   ├── convert_to_rlds.py
│   ├── quantize_gemma4_autoround.py  ← W4A16 양자화 (RoPE 이종 차원 패치 포함) ★
│   ├── gemma4_server.py              ← vLLM 컨테이너용 FastAPI 서버 (비전 지원) ★★
│   ├── run_vllm_server.sh            ← vLLM 컨테이너 서버 시작 ★★
│   ├── build_trtllm_docker.sh        ← TRT-LLM Docker 빌드 ★
│   ├── convert_gemma4_trtllm.sh      ← TRT-LLM 엔진 변환 ★
│   └── run_trtllm_server.sh          ← TRT-LLM 서버 실행 ★
├── data/
│   ├── episodes/              ← HITL 수집 데이터 (attention_source/context 포함)
│   ├── adapters/              ← 학습된 LoRA 어댑터
│   ├── memory/                ← SWM 이력 (swm_history.json)
│   ├── gemma4_autoround_w4a16/ ← auto_round W4A16 RTN 양자화 원본 모델 (Phase 6.7 이전 프로덕션 소스)
│   ├── gemma4_qat_w4a16/      ← Google 공식 QAT 체크포인트 원본 (HF 다운로드, Phase 6.7) ★
│   ├── gemma4_vllm_patched/   ← vLLM 서빙용 패치 모델 = **QAT (현재 프로덕션, Phase 6.7)** ★
│   ├── gemma4_vllm_patched_rtn_backup/ ← 이전 RTN 패치 모델 (롤백용 백업, Phase 6.7)
│   ├── trtllm/engine/         ← TRT-LLM 엔진 (빌드 완료 후 생성)
│   └── trtllm/visual_engine/  ← TRT-LLM 비전 인코더
├── src/
│   ├── haru_vision/           ← 카메라 노드 (듀얼, 3Hz, 336×336 JPEG, Phase 6.3: IMG 축소)
│   ├── haru_audio/            ← 마이크 노드 (VAD, 16kHz, C270)
│   ├── haru_attention/        ← System 3: 주의 노드 (Phase 5.5)
│   │   └── attention_node.py  ← 얼굴감지 + 5-State FSM + VAD통합 + 0.3s 노이즈 필터
│   ├── haru_brain/            ← System 2 (Gemma 4 12B)
│   │   ├── brain_node.py                  ← 스트리밍 콜백 + 조기 발행 + 11관절 로그
│   │   ├── gemma4_inference.py            ← HF bf16 기본 경로 (**kwargs 추가)
│   │   ├── gemma4_trtllm_inference.py     ← vLLM 고속 경로 ★ (streaming, speech/expr 콜백)
│   │   ├── gemma4_autoround_inference.py  ← auto_round W4A16 중속 경로 (**kwargs 추가)
│   │   ├── tom_prompt.py          ← 11관절 전체 + SWM window=2 + 응답 압축
│   │   ├── session_memory.py
│   │   └── adapter_manager.py
│   ├── haru_action/           ← System 1 (50Hz Smoothstep, 9관절+2바퀴)
│   ├── haru_tts/              ← TTS 노드 (edge-tts ko-KR, Phase 6.1 신규) ★
│   ├── haru_expression/       ← 표정 디스플레이 (Robot-display-HRI, 14감정, FULLSCREEN, Phase 6.5.5 교체) ★
│   │   └── haru_expression/
│   │       ├── expression_node.py  ← ROS2 래퍼 (emotion_queue → run_face_app)
│   │       └── robot_face/         ← Robot-display-HRI 소스 (github.com/HGU-SIRLab/Robot-display-HRI)
│   │           ├── main.py         ← RobotFaceApp, FULLSCREEN, 키보드/마우스 제거
│   │           ├── common_helpers.py
│   │           └── emotions/       ← 14 감정 모듈
│   └── haru_logger/           ← HITL 로거 (attention_source/context 저장)
├── scripts/
│   ├── gemma4_mm_patch.py     ← Gemma4UnifiedVisionEmbedder 패치 (Phase 5.8) ★
│   ├── vllm_serve_cmd.sh      ← vLLM serve 명령 (--enable-prefix-caching 포함)
│   ├── train_lora.py          ← QLoRA (11관절 _ALL_JOINTS 학습 타겟)
│   └── ...
├── torch-2.5.0a0+...nv24.08...whl  ← Jetson 전용 torch wheel (사용됨)
├── launch_vla.sh              ← 실행 스크립트 (7-node all / hitl / brain 등)
├── README.md                  ← 영문 전체 가이드 (Phase 6.3 업데이트)
├── HARU_PROJECT_CONTEXT.md    ← 이 파일
├── HARU_RESEARCH_GOALS.txt
└── HARU_RUN_COMMANDS.txt
```
