import os
import yaml
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "kb": {
        "storage_dir": "./kb_storage",
        "notes_dir": "./data/notes",
    },
    "embedding": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu",
    },
    "llm": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
    },
    "chunking": {
        "max_tokens": 500,
        "overlap": 50,
    },
    "retrieval": {
        "top_k": 3,
    },
}


class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self._data = DEFAULT_CONFIG.copy()
        path = Path(config_path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                user = yaml.safe_load(f) or {}
                self._deep_merge(self._data, user)
        self._resolve_env(self._data)

    def _deep_merge(self, base: dict, override: dict) -> None:
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

    def _resolve_env(self, d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                self._resolve_env(v)
            elif isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                d[k] = os.environ.get(v[2:-1], "")

    def get(self, *keys: str) -> Any:
        d = self._data
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k)
            else:
                return None
        return d
