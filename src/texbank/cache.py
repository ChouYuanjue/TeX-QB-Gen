import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from .config import get_pipeline_config


class DiskCache:
    def __init__(self, namespace: str = "llm"):
        cfg = get_pipeline_config().cache
        self.root = Path(cfg.root) / namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = cfg.ttl_seconds

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def _is_expired(self, path: Path) -> bool:
        if self.ttl is None:
            return False
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return True
        return (time.time() - mtime) > self.ttl

    def load(self, key: str) -> Optional[Any]:
        path = self._path(key)
        if not path.exists() or self._is_expired(path):
            return None
        try:
            with path.open('r', encoding='utf-8') as fh:
                return json.load(fh)
        except Exception:
            return None

    def save(self, key: str, value: Any) -> None:
        path = self._path(key)
        tmp = path.with_suffix('.tmp')
        try:
            with tmp.open('w', encoding='utf-8') as fh:
                json.dump(value, fh, ensure_ascii=False, indent=2)
            tmp.replace(path)
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    @staticmethod
    def make_key(*parts: str) -> str:
        data = "|".join(parts)
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
