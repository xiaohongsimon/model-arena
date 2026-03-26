from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import yaml


REQUIRED_FIELDS = ["endpoint", "model_name", "api_key", "timeout_s", "max_input_tokens", "max_output_tokens", "status"]
VALID_STATUSES = {"active", "retired"}


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    endpoint: str
    model_name: str
    api_key: str
    timeout_s: int
    max_input_tokens: int
    max_output_tokens: int
    status: str


_cache: dict[str, tuple[float, dict[str, ModelConfig]]] = {}


def load_config(path: str, active_only: bool = False) -> dict[str, ModelConfig]:
    mtime = os.path.getmtime(path)
    if path in _cache and _cache[path][0] == mtime:
        cached = _cache[path][1]
        if active_only:
            return {k: v for k, v in cached.items() if v.status == "active"}
        return cached

    with open(path) as f:
        raw = yaml.safe_load(f)

    models: dict[str, ModelConfig] = {}
    for model_id, cfg in raw.get("models", {}).items():
        models[model_id] = ModelConfig(
            model_id=model_id,
            endpoint=cfg["endpoint"],
            model_name=cfg["model_name"],
            api_key=cfg["api_key"],
            timeout_s=cfg["timeout_s"],
            max_input_tokens=cfg["max_input_tokens"],
            max_output_tokens=cfg["max_output_tokens"],
            status=cfg["status"],
        )

    _cache[path] = (mtime, models)
    if active_only:
        return {k: v for k, v in models.items() if v.status == "active"}
    return models


def validate_config(path: str) -> list[str]:
    with open(path) as f:
        raw = yaml.safe_load(f)

    errors: list[str] = []
    for model_id, cfg in raw.get("models", {}).items():
        if not isinstance(cfg, dict):
            errors.append(f"{model_id}: config must be a dict")
            continue
        for field in REQUIRED_FIELDS:
            if field not in cfg:
                errors.append(f"{model_id}: missing required field '{field}'")
        if "status" in cfg and cfg["status"] not in VALID_STATUSES:
            errors.append(f"{model_id}: status must be one of {VALID_STATUSES}, got '{cfg['status']}'")
    return errors
