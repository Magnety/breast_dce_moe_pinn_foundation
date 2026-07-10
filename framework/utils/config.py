from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or validated."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve(strict=False)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ConfigError("PyYAML is required to read YAML config files.") from exc

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level config must be a mapping: {config_path}")
    data = expand_environment(data)
    data["_config_path"] = str(config_path)
    return data


def deep_get(mapping: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def expand_environment(value: Any) -> Any:
    """Recursively expand environment variables in config strings."""

    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_environment(item) for item in value)
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    return value
