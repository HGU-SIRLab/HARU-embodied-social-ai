"""
HARU LoRA 어댑터 관리자

data/adapters/ 아래 QLoRA 학습으로 생성된 어댑터 디렉토리를 관리.
현재는 가장 최근 어댑터를 자동 선택 (S-LoRA 컨텍스트 라우팅은 Phase 6).

어댑터 디렉토리 구조:
  data/adapters/
    adapter_20260618_143000/
      adapter_config.json
      adapter_model.safetensors (or pytorch_model.bin)
      README.md (에피소드 수, 학습 날짜 등)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_ADAPTERS_DIR = Path('/home/herobot/robot_brain_workspace/data/adapters')


def list_adapters() -> list[Path]:
    """학습된 어댑터 디렉토리 목록 (생성 시간 오름차순)."""
    if not _ADAPTERS_DIR.exists():
        return []
    dirs = sorted(
        [d for d in _ADAPTERS_DIR.iterdir()
         if d.is_dir() and (d / 'adapter_config.json').exists()],
        key=lambda d: d.stat().st_mtime,  # 이름 정렬보다 실제 생성 시간 기준이 안전
    )
    return dirs


def latest_adapter() -> Path | None:
    """가장 최근 어댑터 경로 반환. 없으면 None."""
    adapters = list_adapters()
    if not adapters:
        logger.info('[Adapter] 사용 가능한 어댑터 없음 (베이스 모델로 동작)')
        return None
    path = adapters[-1]
    logger.info(f'[Adapter] 최신 어댑터: {path.name}')
    return path


def adapter_info(adapter_path: Path) -> dict:
    """어댑터 메타데이터 반환 (없으면 빈 dict)."""
    readme = adapter_path / 'README.md'
    if readme.exists():
        return {'readme': readme.read_text(encoding='utf-8')}
    cfg = adapter_path / 'adapter_config.json'
    if cfg.exists():
        try:
            return json.loads(cfg.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {}
