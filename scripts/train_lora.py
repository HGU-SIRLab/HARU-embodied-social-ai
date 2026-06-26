#!/usr/bin/env python3
"""
HARU Phase 5 — HITL 에피소드 QLoRA 파인튜닝

HITL로 수집한 에피소드를 Supervised Fine-Tuning으로 학습하여
LoRA 어댑터 생성. 베이스 모델(Gemma 4 12B)은 고정, LoRA 파라미터만 업데이트.

사용법:
  python scripts/train_lora.py                         # 기본 설정
  python scripts/train_lora.py --epochs 3 --rank 16
  python scripts/train_lora.py --dry-run               # 데이터 파싱만 확인

생성 결과:
  data/adapters/adapter_YYYYMMDD_HHMMSS/
    adapter_config.json
    adapter_model.safetensors
    README.md
"""

import sys
_VENV_SITE = '/home/herobot/robot_brain_workspace/haru_vla_env/lib/python3.10/site-packages'
if _VENV_SITE not in sys.path:
    sys.path.insert(0, _VENV_SITE)

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

WORKSPACE    = Path('/home/herobot/robot_brain_workspace')
EPISODES_DIR = WORKSPACE / 'data' / 'episodes'
ADAPTERS_DIR = WORKSPACE / 'data' / 'adapters'
MODEL_ID     = 'google/gemma-4-12B-it'

# 패키지 경로 등록
sys.path.insert(0, str(WORKSPACE / 'src' / 'haru_logger'))
sys.path.insert(0, str(WORKSPACE / 'src' / 'haru_brain'))
from haru_logger.episode_writer import normalized_to_action  # noqa: E402
from haru_brain.tom_prompt import SYSTEM_PROMPT              # noqa: E402


# ── 에피소드 로더 ──────────────────────────────────────────────────────────────

def load_episodes(episodes_dir: Path) -> list[dict]:
    samples = []
    if not episodes_dir.exists():
        logger.warning(f'에피소드 디렉토리 없음: {episodes_dir}')
        return samples

    ep_dirs = sorted(d for d in episodes_dir.iterdir() if d.is_dir())
    logger.info(f'에피소드 디렉토리 {len(ep_dirs)}개 발견')

    for ep_dir in ep_dirs:
        for step_file in sorted(ep_dir.glob('step_*.npz')):
            try:
                data = np.load(step_file, allow_pickle=False)
                def _to_str(x) -> str:
                    return x.decode('utf-8') if isinstance(x, bytes) else str(x)
                samples.append({
                    'image':     data['image'],        # (H, W, 3) uint8
                    'action':    data['action'],        # 12-dim float32 정규화
                    'speech':    _to_str(data['speech_text'].item()),
                    'emotion':   _to_str(data['emotion'].item()),
                    'corrected': bool(data['is_corrected'].item()),
                    'source':    str(step_file),
                })
            except Exception as e:
                logger.warning(f'스텝 로드 실패 {step_file}: {e}')

    n_corr = sum(s['corrected'] for s in samples)
    logger.info(f'총 스텝 {len(samples)}개 로드 (교정됨: {n_corr}개)')
    return samples


# episode_writer의 long name → inference 약어 매핑
_JOINT_ABBREV = {
    'head_tilt': 'ht', 'head_pan': 'hp', 'head_roll': 'hr',
    'r_arm_pitch': 'rap', 'r_shoulder_roll': 'rsr', 'r_elbow_pitch': 'rep',
    'l_arm_pitch': 'lap', 'l_shoulder_roll': 'lsr', 'l_elbow_pitch': 'lep',
    'right_wheel': 'rw',  'left_wheel': 'lw',
}


def _build_target_json(sample: dict) -> str:
    """샘플 → 모델이 생성해야 할 JSON 문자열 (inference 포맷과 동일)."""
    action_dict = normalized_to_action(sample['action'])
    expr_id = int(np.clip(round(action_dict.get('expression_id', 0)), 0, 7))
    # long name → 약어 변환 (inference 포맷 일치)
    action_abbrev = {_JOINT_ABBREV[k]: round(v, 1)
                     for k, v in action_dict.items()
                     if k in _JOINT_ABBREV}
    return json.dumps({
        'speech':        sample['speech'],
        'emotion':       sample['emotion'],
        'expression_id': expr_id,
        'action':        action_abbrev,
    }, ensure_ascii=False)


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class HaruEpisodeDataset(Dataset):
    """HITL 에피소드를 processor로 사전 토큰화하여 캐싱.

    loss 마스킹: 프롬프트(system+user) 구간은 -100, 어시스턴트 응답만 학습.
    PIL Image는 processor 처리 후 pixel_values 텐서로 변환해 저장.
    """

    def __init__(self, samples: list[dict], processor):
        self._items: list[dict] = []
        fail = 0
        for s in samples:
            item = self._preprocess(s, processor)
            if item is not None:
                self._items.append(item)
            else:
                fail += 1
        logger.info(f'Dataset: {len(self._items)}개 준비 완료 (실패 {fail}개)')

    @staticmethod
    def _preprocess(sample: dict, processor) -> dict | None:
        pil_img     = Image.fromarray(sample['image'])
        target_json = _build_target_json(sample)

        msgs_prompt = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': [
                {'type': 'image', 'image': pil_img},
                {'type': 'text',  'text': '현재 상황을 관찰하고 HARU로서 응답하세요.'},
            ]},
        ]
        msgs_full = msgs_prompt + [{'role': 'assistant', 'content': target_json}]

        try:
            # 프롬프트만 토큰화 → 어시스턴트 응답 시작 위치 파악
            prompt_enc = processor.apply_chat_template(
                msgs_prompt,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors='pt',
            )
            # 전체 대화 토큰화 (pixel_values 포함)
            full_enc = processor.apply_chat_template(
                msgs_full,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors='pt',
            )
        except Exception as e:
            logger.warning(f'토큰화 실패 ({sample["source"]}): {e}')
            return None

        input_ids = full_enc['input_ids'][0]          # (seq_len,)
        attn_mask = full_enc['attention_mask'][0]
        prompt_len = prompt_enc['input_ids'].shape[1]

        # 어시스턴트 응답 구간만 loss 계산 (프롬프트 → -100)
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        item = {
            'input_ids':      input_ids,
            'attention_mask': attn_mask,
            'labels':         labels,
        }
        if 'pixel_values' in full_enc and full_enc['pixel_values'] is not None:
            item['pixel_values'] = full_enc['pixel_values'][0]

        return item

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def collate_fn(batch: list[dict]) -> dict:
    """batch_size=1 전용 collator: 시퀀스 축 추가만 수행."""
    item = batch[0]
    out = {
        'input_ids':      item['input_ids'].unsqueeze(0),
        'attention_mask': item['attention_mask'].unsqueeze(0),
        'labels':         item['labels'].unsqueeze(0),
    }
    if 'pixel_values' in item:
        out['pixel_values'] = item['pixel_values'].unsqueeze(0)
    return out


# ── 학습 루프 ──────────────────────────────────────────────────────────────────

def train(
    samples:  list[dict],
    rank:     int   = 16,
    alpha:    int   = 32,
    epochs:   int   = 3,
    lr:       float = 2e-4,
    grad_acc: int   = 4,
) -> Path:
    from transformers import AutoProcessor, Gemma4UnifiedForConditionalGeneration
    from transformers import get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model, TaskType

    logger.info('[Train] 베이스 모델 로드 (bf16)...')
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # gradient_checkpointing use_reentrant=False: Gemma 4 KV 공유 레이어 버그 우회
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    logger.info('[Train] 데이터셋 사전 토큰화 중...')
    dataset = HaruEpisodeDataset(samples, processor)
    if len(dataset) == 0:
        raise RuntimeError('사전 토큰화 후 유효한 샘플 없음. 에피소드 데이터를 확인하세요.')

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    # 데이터셋이 작아도 total_steps=0이 되지 않도록 보호
    total_steps = max(1, len(loader) * epochs // grad_acc)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    device = next(model.parameters()).device
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for local_step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / grad_acc
            loss.backward()
            epoch_loss += outputs.loss.item()

            if local_step % grad_acc == 0 or local_step == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 5 == 0:
                    logger.info(
                        f'[Train] epoch {epoch}/{epochs}  step {global_step}  '
                        f'loss {outputs.loss.item():.4f}'
                    )

        avg = epoch_loss / len(loader)
        logger.info(f'[Train] epoch {epoch} 완료 — avg loss: {avg:.4f}')

    # LoRA 어댑터만 저장 (베이스 모델 가중치 제외)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = ADAPTERS_DIR / f'adapter_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    logger.info(f'[Train] 어댑터 저장 완료: {out_dir}')

    readme = (
        f'# HARU LoRA Adapter — {ts}\n\n'
        f'- 학습 스텝: {len(samples)}\n'
        f'- 교정됨: {sum(s["corrected"] for s in samples)}개\n'
        f'- Epoch: {epochs}\n'
        f'- LoRA rank: {rank}, alpha: {alpha}, lr: {lr}\n'
        f'- 베이스 모델: {MODEL_ID}\n'
    )
    (out_dir / 'README.md').write_text(readme, encoding='utf-8')
    return out_dir


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='HARU Phase 5 QLoRA 파인튜닝')
    parser.add_argument('--episodes', type=Path,  default=EPISODES_DIR)
    parser.add_argument('--epochs',   type=int,   default=3)
    parser.add_argument('--rank',     type=int,   default=16)
    parser.add_argument('--alpha',    type=int,   default=32)
    parser.add_argument('--lr',       type=float, default=2e-4)
    parser.add_argument('--grad-acc', type=int,   default=4, dest='grad_acc')
    parser.add_argument('--dry-run',  action='store_true')
    args = parser.parse_args()

    samples = load_episodes(args.episodes)
    if not samples:
        logger.error('학습 데이터 없음. HITL로 에피소드를 먼저 수집하세요: ./launch_vla.sh hitl')
        sys.exit(1)

    if args.dry_run:
        logger.info('[Dry-run] 데이터 파싱 성공.')
        logger.info(f'[Dry-run] 샘플 0 target_json:\n{_build_target_json(samples[0])}')
        return

    out = train(samples, rank=args.rank, alpha=args.alpha,
                epochs=args.epochs, lr=args.lr, grad_acc=args.grad_acc)
    logger.info(f'[Done] 완료: {out}')
    logger.info('[Done] 다음 brain_node 실행 시 어댑터가 자동 로드됩니다.')


if __name__ == '__main__':
    main()
