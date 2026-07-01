#!/usr/bin/env python3
"""
scripts/validate_rope_ppl.py
RoPE 패치 수치 검증 — Perplexity 비교

검증 구조:
  [A] bf16 원본        — 패치 없음 (정답 기준선)
  [B] bf16 + RoPE 패치 — 패치만 적용, 양자화 없음 (패치 단독 영향)
  [C] W4A16 + RoPE 패치 — 실제 운용 환경 (패치 + 양자화 통합)

비교 해석:
  B - A = 순수 패치 오차 (이 값이 작아야 패치가 수치적으로 안전)
  C - A = 패치 + 양자화 통합 오차 (5% 이내 권장)
  C - B = 순수 양자화 오차 (참고용)

Jetson 메모리 제약으로 단계별 분리 실행:
  source haru_vla_env/bin/activate
  python scripts/validate_rope_ppl.py --mode bf16       # ~22GB, 30분
  python scripts/validate_rope_ppl.py --mode bf16_patch # ~22GB, 30분
  python scripts/validate_rope_ppl.py --mode w4a16      # ~7.4GB, 20분
  python scripts/validate_rope_ppl.py --mode report     # 결과 비교
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

# ── 경로 설정 ────────────────────────────────────────────────────────────────
WORKSPACE   = Path('/home/herobot/robot_brain_workspace')
HF_MODEL    = Path('/home/herobot/.cache/huggingface/hub/'
                   'models--google--gemma-4-12B-it/snapshots/'
                   '5926caa4ec0cac5cbfadaf4077420520de1d5205')
W4A16_DIR   = WORKSPACE / 'data' / 'gemma4_autoround_w4a16'
RESULTS_FILE = Path('/tmp/haru_ppl_results.json')

# ── 검증용 텍스트 ────────────────────────────────────────────────────────────
# Gemma 4 12B-it 은 instruction-tuned 모델이므로 chat template 형식으로 입력해야
# 정상적인 PPL이 나옴. _build_texts()에서 apply_chat_template으로 래핑.
_RAW_TEXTS = [
    # 사회적 상호작용 — HARU 도메인
    "안녕하세요! 오늘 기분이 어떠세요?",
    "많이 힘드셨나요? 무슨 일이 있었는지 이야기해 주실 수 있나요?",
    "걱정하지 마세요. 제가 항상 곁에 있을게요.",
    "오늘 정말 수고하셨어요. 잠깐 쉬어 가세요.",
    "요즘 잠을 잘 못 주무시나요? 얼굴이 피곤해 보여서요.",
    "오늘 좋은 일이 있었나요? 표정이 밝아 보여요.",
    "혼자 있으니까 외롭지 않으세요? 저랑 이야기해요.",
    "무엇이 그렇게 고민이세요? 같이 생각해 볼까요?",
    # 일반 한국어
    "인공지능은 인간의 감정을 진정으로 이해할 수 있을까요?",
    "로봇과 인간이 함께 생활하는 미래가 어떤 모습일지 상상해 보세요.",
    "서울의 봄은 벚꽃이 아름답게 피어 도시 전체를 물들입니다.",
    "한국의 전통 음식인 비빔밥은 다양한 재료가 조화를 이루는 요리입니다.",
    "매일 꾸준히 운동하면 건강을 유지하는 데 큰 도움이 됩니다.",
    "좋은 친구란 기쁠 때나 슬플 때나 항상 함께해 주는 사람입니다.",
    "새로운 기술을 배우는 것은 어렵지만 그만큼 보람도 큽니다.",
    "독서는 마음의 양식이라는 말처럼 책 속에는 무한한 세계가 있습니다.",
    # 감정 표현
    "기쁨과 슬픔, 분노와 평온함이 교차하는 하루를 보냈습니다.",
    "갑작스러운 소식에 놀라서 말문이 막혔습니다.",
    "오랜만에 만난 친구와 이야기하니 마음이 따뜻해졌습니다.",
    "혼자 해결해야 한다는 압박감에 두려움을 느꼈습니다.",
    # 영어
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Artificial intelligence has transformed how we interact with machines.",
    "Social robots are designed to engage with humans in natural ways.",
    "The ability to understand and respond to human emotions is key.",
    "Language models learn patterns from vast amounts of text data.",
    "Empathy is a fundamental human trait that robots strive to replicate.",
    "The future of human-robot interaction depends on trust and understanding.",
    "Neural networks process information in layers of increasing abstraction.",
    # 복합 문장
    ("인간-로봇 상호작용 연구는 로봇이 인간의 감정을 인식하고 적절하게 반응할 수 있도록 "
     "하는 기술 개발에 집중하고 있으며, 이를 통해 노인 돌봄, 교육, 의료 분야에서 "
     "혁신적인 변화가 기대됩니다."),
    ("The development of large language models has accelerated the progress of "
     "natural language processing, enabling machines to generate coherent and "
     "contextually appropriate responses to complex human queries."),
]


def _build_texts(tokenizer) -> list[str]:
    """
    chat template을 적용한 전체 문자열 반환.
    Gemma 4-it은 <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n 형식으로
    학습됐으므로 raw text 그대로 넣으면 PPL이 비정상적으로 높게 나옴.
    """
    out = []
    for text in _RAW_TEXTS:
        try:
            chat = [{"role": "user", "content": text}]
            formatted = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True)
            out.append(formatted)
        except Exception:
            out.append(text)  # fallback: raw text
    return out


# ── RoPE 패치 함수 ────────────────────────────────────────────────────────────
def apply_rope_patch():
    """quantize_gemma4_autoround.py와 동일한 RoPE 패치 적용."""
    try:
        import transformers.models.gemma4_unified.modeling_gemma4_unified as _g4u_mod
    except ImportError:
        print("[경고] gemma4_unified 모듈을 찾을 수 없습니다. 패치 건너뜀.")
        return False

    def _safe_apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
        """head_dim 불일치 시 RoPE 적용 가능한 차원만 회전, 나머지 pass-through."""
        if unsqueeze_dim is not None:
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
        rot_dim = min(x.shape[-1], cos.shape[-1])
        if rot_dim < x.shape[-1]:
            x_rot  = x[..., :rot_dim]
            x_pass = x[..., rot_dim:]
            x_rot  = (x_rot * cos[..., :rot_dim]) + (
                _g4u_mod.rotate_half(x_rot) * sin[..., :rot_dim])
            return torch.cat([x_rot, x_pass], dim=-1)
        return (x * cos) + (_g4u_mod.rotate_half(x) * sin)

    _g4u_mod.apply_rotary_pos_emb = _safe_apply_rotary_pos_emb
    print("[패치] RoPE 이종 차원 패치 적용 완료 (min-dim truncation)")
    return True


# ── 모델 로드 함수 ────────────────────────────────────────────────────────────
def _common_load(model_path: Path, label: str, w4a16: bool = False):
    """bf16/W4A16 공통 로드. AutoProcessor 우선, 없으면 AutoTokenizer 사용."""
    from transformers import AutoProcessor, AutoTokenizer
    try:
        from transformers import Gemma4UnifiedForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import Gemma4ForConditionalGeneration as ModelClass

    print(f"[로드] {label}: {model_path}")

    # Gemma 4 Unified는 AutoProcessor가 올바른 chat template을 포함
    try:
        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=True)
        tokenizer = processor.tokenizer
        print("[로드] AutoProcessor → tokenizer 추출 완료")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True)
        print("[로드] AutoTokenizer 사용 (fallback)")

    model = ModelClass.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map='auto',
        local_files_only=True,
        trust_remote_code=True,
    )

    if w4a16:
        try:
            from auto_round.inference.convert_model import convert_hf_model, post_init
            model, used_backends = convert_hf_model(model, target_device='cuda')
            post_init(model, used_backends)
            print(f"[로드] QuantLinear 변환 완료 (backends={used_backends})")
        except Exception as e:
            print(f"[경고] convert_hf_model 실패 (W4A16 weights가 올바르지 않을 수 있음): {e}")

    model.eval()
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"[로드] 완료. GPU 메모리: {mem_gb:.1f} GB")
    return model, tokenizer


def load_bf16_model(model_path: Path):
    return _common_load(model_path, 'bf16 원본', w4a16=False)


def load_w4a16_model(model_path: Path):
    return _common_load(model_path, 'W4A16 패치', w4a16=True)


# ── Perplexity 계산 ───────────────────────────────────────────────────────────
def calc_perplexity(model, tokenizer, texts: list[str],
                    max_length: int = 512) -> dict:
    """
    각 텍스트의 perplexity를 계산.
    - chat template이 적용된 전체 문자열을 입력으로 사용
    - 레이블: input_ids 그대로 (pad 없으므로 마스킹 불필요)
    - max_length: 512 (chat template 래핑 후 길이 증가 고려)
    """
    model.eval()
    results = []
    total_nll = 0.0
    total_tokens = 0

    print(f"\n[PPL] {len(texts)}개 문장 계산 시작 (chat template 적용됨)...")
    for i, text in enumerate(texts):
        try:
            enc = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,  # chat template이 이미 special token 포함
            ).to('cuda')

            input_ids = enc['input_ids']
            n_tokens = input_ids.size(1)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss.item()

            ppl = math.exp(min(loss, 20))  # exp(20)≈485M 이상은 overflow 방지용 cap
            total_nll += loss * n_tokens
            total_tokens += n_tokens
            results.append({'text_idx': i, 'ppl': ppl, 'loss': round(loss, 6),
                            'n_tokens': n_tokens})
            raw = _RAW_TEXTS[i] if i < len(_RAW_TEXTS) else text
            print(f"  [{i+1:2d}/{len(texts)}] loss={loss:.4f}  PPL={ppl:8.2f}"
                  f"  tok={n_tokens:3d}  {raw[:35]}...")

        except Exception as e:
            print(f"  [{i+1:2d}/{len(texts)}] 오류: {e}")
            results.append({'text_idx': i, 'ppl': None, 'error': str(e)})

    valid = [r for r in results if r.get('ppl') is not None]
    corpus_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else None
    mean_ppl   = sum(r['ppl'] for r in valid) / len(valid) if valid else None
    ppls       = sorted(r['ppl'] for r in valid)
    median_ppl = ppls[len(ppls)//2] if ppls else None

    print(f"\n[PPL] 코퍼스 PPL (전체 weighted): {corpus_ppl:.4f}")
    print(f"[PPL] 문장별 평균 PPL:            {mean_ppl:.4f}")
    print(f"[PPL] 문장별 중앙값 PPL:           {median_ppl:.4f}")
    print(f"[PPL] 유효 문장: {len(valid)}/{len(texts)}, 총 토큰: {total_tokens}")

    return {
        'corpus_ppl': corpus_ppl,
        'mean_ppl': mean_ppl,
        'median_ppl': median_ppl,
        'total_tokens': total_tokens,
        'n_valid': len(valid),
        'per_text': results,
    }


# ── 결과 저장/로드 ────────────────────────────────────────────────────────────
def load_results() -> dict:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def save_result(mode: str, data: dict):
    results = load_results()
    results[mode] = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), **data}
    RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n[저장] 결과 저장 완료: {RESULTS_FILE}")


# ── 비교 리포트 ───────────────────────────────────────────────────────────────
def print_report():
    results = load_results()
    if not results:
        print("[리포트] 저장된 결과 없음. --mode bf16 부터 실행하세요.")
        return

    print("\n" + "="*65)
    print("  HARU RoPE 패치 수치 검증 리포트")
    print("="*65)

    rows = [
        ('bf16',       '[A] bf16 원본 (기준선)',            None),
        ('bf16_patch', '[B] bf16 + RoPE 패치',             'bf16'),
        ('w4a16',      '[C] W4A16 + RoPE 패치 (운용 환경)', 'bf16'),
    ]

    ref_corpus = results.get('bf16', {}).get('corpus_ppl')

    for mode, label, ref_mode in rows:
        r = results.get(mode)
        if r is None:
            print(f"\n  {label}")
            print(f"    → 아직 실행 안 됨 (python validate_rope_ppl.py --mode {mode})")
            continue

        c_ppl = r.get('corpus_ppl', 'N/A')
        m_ppl = r.get('mean_ppl',   'N/A')
        ts    = r.get('timestamp',   '?')

        print(f"\n  {label}  [{ts}]")
        print(f"    코퍼스 PPL: {c_ppl:.4f}" if isinstance(c_ppl, float) else f"    코퍼스 PPL: {c_ppl}")
        print(f"    평균   PPL: {m_ppl:.4f}" if isinstance(m_ppl, float) else f"    평균   PPL: {m_ppl}")

        if ref_mode and ref_corpus and isinstance(c_ppl, float):
            diff_pct = (c_ppl - ref_corpus) / ref_corpus * 100
            verdict  = ('✅ 안전 (<5%)' if abs(diff_pct) < 5
                        else '⚠️  주의 (5~15%)' if abs(diff_pct) < 15
                        else '❌ 위험 (>15%)')
            print(f"    기준선 대비: {diff_pct:+.2f}%  {verdict}")

    # 패치 단독 영향 분리
    a = results.get('bf16', {}).get('corpus_ppl')
    b = results.get('bf16_patch', {}).get('corpus_ppl')
    c = results.get('w4a16', {}).get('corpus_ppl')

    if a and b and c:
        print("\n" + "-"*65)
        print("  오차 분해")
        print("-"*65)
        patch_err = (b - a) / a * 100
        quant_err = (c - b) / b * 100
        total_err = (c - a) / a * 100
        print(f"  순수 패치 오차  (B-A): {patch_err:+.2f}%")
        print(f"  순수 양자화 오차(C-B): {quant_err:+.2f}%")
        print(f"  통합 오차       (C-A): {total_err:+.2f}%")

        if abs(patch_err) < 0.5:
            print("\n  → 패치 자체는 수치적으로 거의 무영향 (논문 서술 가능)")
        elif abs(patch_err) < 2.0:
            print("\n  → 패치 오차가 미미함. 'empirically negligible' 로 서술 가능")
        else:
            print("\n  → 패치 오차가 의미 있음. Level 2 (활성화 비교) 검증 권장")

    print("\n" + "="*65)
    print(f"  전체 결과 파일: {RESULTS_FILE}")
    print("="*65 + "\n")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='HARU RoPE 패치 PPL 검증')
    parser.add_argument('--mode', required=True,
                        choices=['bf16', 'bf16_patch', 'w4a16', 'report'],
                        help='실행 모드')
    parser.add_argument('--max-length', type=int, default=256,
                        help='입력 최대 토큰 수 (기본 256)')
    args = parser.parse_args()

    # ── 리포트 모드 ──────────────────────────────────────────────────────────
    if args.mode == 'report':
        print_report()
        return

    # ── 추론 환경 활성화 ─────────────────────────────────────────────────────
    sys.path.insert(0, str(WORKSPACE / 'src' / 'haru_brain'))
    os.environ.setdefault('HARU_QUANTIZED_MODEL_DIR', str(W4A16_DIR))

    print(f"\n{'='*55}")
    print(f"  모드: {args.mode}")
    print(f"  텍스트 수: {len(_RAW_TEXTS)} (chat template 적용)")
    print(f"  최대 토큰: {args.max_length}")
    print(f"{'='*55}\n")

    # ── [A] bf16 원본 ─────────────────────────────────────────────────────────
    if args.mode == 'bf16':
        if not HF_MODEL.exists():
            print(f"[오류] HF 모델 경로 없음: {HF_MODEL}")
            sys.exit(1)
        model, tokenizer = load_bf16_model(HF_MODEL)
        texts = _build_texts(tokenizer)
        stats = calc_perplexity(model, tokenizer, texts, args.max_length)
        save_result('bf16', stats)

    # ── [B] bf16 + RoPE 패치 ─────────────────────────────────────────────────
    elif args.mode == 'bf16_patch':
        if not HF_MODEL.exists():
            print(f"[오류] HF 모델 경로 없음: {HF_MODEL}")
            sys.exit(1)
        apply_rope_patch()
        model, tokenizer = load_bf16_model(HF_MODEL)
        texts = _build_texts(tokenizer)
        stats = calc_perplexity(model, tokenizer, texts, args.max_length)
        save_result('bf16_patch', stats)

    # ── [C] W4A16 + RoPE 패치 ────────────────────────────────────────────────
    elif args.mode == 'w4a16':
        if not W4A16_DIR.exists():
            print(f"[오류] W4A16 모델 경로 없음: {W4A16_DIR}")
            print("       quantize_gemma4_autoround.py 를 먼저 실행하세요.")
            sys.exit(1)
        apply_rope_patch()
        model, tokenizer = load_w4a16_model(W4A16_DIR)
        texts = _build_texts(tokenizer)
        stats = calc_perplexity(model, tokenizer, texts, args.max_length)
        save_result('w4a16', stats)


if __name__ == '__main__':
    main()
