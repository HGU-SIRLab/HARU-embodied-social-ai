"""
에피소드 데이터 저장 모듈

저장 구조:
  data/episodes/
    episode_YYYYMMDD_HHMMSS/
      metadata.json
      step_0000.npz
      step_0001.npz
      ...

각 .npz 키:
  image              : (H, W, 3) uint8
  action             : 12-dim float32 [-1,1] 정규화 (최종/교정값)
  action_vla         : 12-dim float32 [-1,1] 정규화 (brain 원본 예측)
  is_corrected       : bool
  language_instruction: bytes
  speech_text        : bytes
  emotion            : bytes
  attention_source   : bytes (FSM 상태: CONVERSING / APPEARED_TO_SILENT 등)
  attention_context  : bytes (상황 컨텍스트 문자열)
"""

import json
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path

# HARU 12-DoF (표정 1 + 위치 관절 9 + 바퀴 2)
HARU_DOF_RANGES = [
    ("expression_id",   0,     7   ),
    ("head_tilt",       1500,  3086),
    ("head_pan",        1043,  3071),
    ("head_roll",       1630,  2452),
    ("r_arm_pitch",     1024,  2451),
    ("l_arm_pitch",     37,    1542),
    ("r_shoulder_roll", 1000,  2050),
    ("r_elbow_pitch",   2047,  3062),
    ("l_shoulder_roll", 1047,  2056),
    ("l_elbow_pitch",   1021,  2007),
    ("right_wheel",    -300,    300),
    ("left_wheel",     -300,    300),
]

DOF = len(HARU_DOF_RANGES)  # 12

EXPRESSION_LABELS = {
    0: "neutral", 1: "joy",      2: "sadness",  3: "curiosity",
    4: "surprise", 5: "empathy", 6: "thinking", 7: "concern",
}

# brain이 약어(ht/hp/hr/...)로 action을 생성하므로 긴 이름으로 매핑
_ABBREV_TO_LONG = {
    'ht': 'head_tilt',       'hp': 'head_pan',         'hr': 'head_roll',
    'rap': 'r_arm_pitch',    'lap': 'l_arm_pitch',
    'rsr': 'r_shoulder_roll','rep': 'r_elbow_pitch',
    'lsr': 'l_shoulder_roll','lep': 'l_elbow_pitch',
    'rw': 'right_wheel',     'lw': 'left_wheel',
}


def action_to_normalized(action_dict: dict) -> np.ndarray:
    """HaruAction dict → [-1, 1] 정규화 벡터 (12-dim). 약어·긴 이름 모두 허용."""
    expanded = {_ABBREV_TO_LONG.get(k, k): v for k, v in action_dict.items()}
    result = []
    for name, lo, hi in HARU_DOF_RANGES:
        val = float(expanded.get(name, (lo + hi) / 2.0))
        norm = 2.0 * (val - lo) / (hi - lo) - 1.0
        result.append(float(np.clip(norm, -1.0, 1.0)))
    return np.array(result, dtype=np.float32)


def normalized_to_action(norm_vec: np.ndarray) -> dict:
    """[-1, 1] 벡터 → HaruAction dict."""
    result = {}
    for i, (name, lo, hi) in enumerate(HARU_DOF_RANGES):
        result[name] = float((norm_vec[i] + 1.0) / 2.0 * (hi - lo) + lo)
    return result


class EpisodeWriter:
    def __init__(self, base_dir: str = '/home/herobot/robot_brain_workspace/data/episodes'):
        self.base_dir = Path(base_dir)
        self.episode_dir: Path | None = None
        self.step_count = 0
        self.episode_id = 0
        self._count_existing_episodes()

    def _count_existing_episodes(self):
        if self.base_dir.exists():
            self.episode_id = len([
                d for d in self.base_dir.iterdir() if d.is_dir()
            ])

    def start_episode(self) -> str:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.episode_dir = self.base_dir / f'episode_{ts}'
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.step_count = 0
        self.episode_id += 1
        return str(self.episode_dir)

    def save_step(
        self,
        image: np.ndarray,
        vla_action: dict,
        final_action: dict,
        language_instruction: str,
        speech_text: str,
        emotion: str,
        is_corrected: bool,
        attention_source: str = '',
        attention_context: str = '',
    ) -> int:
        if self.episode_dir is None:
            raise RuntimeError('start_episode() 먼저 호출하세요.')

        vla_norm   = action_to_normalized(vla_action)
        final_norm = action_to_normalized(final_action)

        step_path = self.episode_dir / f'step_{self.step_count:04d}.npz'
        np.savez_compressed(
            step_path,
            image=image,
            action=final_norm,
            action_vla=vla_norm,
            is_corrected=np.bool_(is_corrected),
            language_instruction=np.bytes_(language_instruction.encode()),
            speech_text=np.bytes_(speech_text.encode()),
            emotion=np.bytes_(emotion.encode()),
            attention_source=np.bytes_(attention_source.encode()),
            attention_context=np.bytes_(attention_context.encode()),
        )
        idx = self.step_count
        self.step_count += 1
        return idx

    def end_episode(self, accepted: bool = True) -> bool:
        if self.episode_dir is None:
            return False

        if not accepted or self.step_count == 0:
            shutil.rmtree(self.episode_dir, ignore_errors=True)
            self.episode_dir = None
            return False

        metadata = {
            'episode_id':  self.episode_id,
            'timestamp':   datetime.now().isoformat(),
            'total_steps': self.step_count,
            'dof':         DOF,
            'dof_ranges':  {name: [lo, hi] for name, lo, hi in HARU_DOF_RANGES},
        }
        with open(self.episode_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.episode_dir = None
        return True

    @property
    def current_step(self) -> int:
        return self.step_count
