"""
HARU 세션 간 장기 기억 (SWM 영속 레이어)

세션이 끊겨도 과거 상호작용 이력을 디스크에 보존.
시작 시 로드 → 4-turn 추론 윈도우에 공급 → 종료 시 저장.

저장 위치: data/memory/swm_history.json
형식: [{"user": str, "assistant": str, "ts": str}, ...]
최대 저장: KEEP_TURNS 쌍 (초과 시 오래된 것부터 삭제)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_MEMORY_PATH = Path('/home/herobot/robot_brain_workspace/data/memory/swm_history.json')
KEEP_TURNS = 50  # 디스크에 보존할 최대 대화 쌍 수


def load_history() -> list[tuple[str, str]]:
    """저장된 (user_text, assistant_text) 쌍 목록 반환. 없으면 빈 리스트."""
    if not _MEMORY_PATH.exists():
        return []
    try:
        records = json.loads(_MEMORY_PATH.read_text(encoding='utf-8'))
        return [(r['user'], r['assistant']) for r in records if 'user' in r and 'assistant' in r]
    except Exception as e:
        logger.warning(f'[Memory] 이력 로드 실패: {e}')
        return []


def save_history(pairs: list[tuple[str, str]]):
    """(user_text, assistant_text) 쌍 목록을 디스크에 저장 (원자적 쓰기)."""
    _MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    keep = pairs[-KEEP_TURNS:] if len(pairs) > KEEP_TURNS else pairs
    now = datetime.now().isoformat(timespec='seconds')
    records = [{'user': u, 'assistant': a, 'ts': now} for u, a in keep]
    # 임시 파일에 쓴 뒤 rename — 쓰기 도중 프로세스 종료 시 파일 손상 방지
    tmp = _MEMORY_PATH.parent / f'.swm_history_tmp_{_MEMORY_PATH.stat().st_ino if _MEMORY_PATH.exists() else "new"}.json'
    try:
        tmp.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        tmp.replace(_MEMORY_PATH)
    except Exception as e:
        logger.warning(f'[Memory] 이력 저장 실패: {e}')
        tmp.unlink(missing_ok=True)
