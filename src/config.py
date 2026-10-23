"""
Configuration Module.

Responsible for loading, validating, and providing project-wide configuration
settings defined in `config.yaml`.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _fallback_yaml_parser(content: str) -> Dict[str, Any]:
    """
    Minimal fallback parser for simple key-value YAML when PyYAML is not yet installed.
    """
    result: Dict[str, Any] = {}
    current_section = result
    section_stack = [(0, result)]

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))

        while section_stack and indent < section_stack[-1][0]:
            section_stack.pop()

        current_target = section_stack[-1][1] if section_stack else result

        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()

            if val == "":
                new_dict: Dict[str, Any] = {}
                current_target[key] = new_dict
                section_stack.append((indent + 2, new_dict))
            else:
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                elif val.startswith("'") and val.endswith("'"):
                    val = val[1:-1]
                elif val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
                elif val.isdigit():
                    val = int(val)
                else:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                current_target[key] = val

    return result


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Load project configuration from a YAML file.

    Parameters
    ----------
    config_path : Optional[str | Path]
        Path to the configuration YAML file. If None, defaults to `config.yaml`
        in the project root.

    Returns
    -------
    Dict[str, Any]
        Dictionary representation of configuration settings.

    Raises
    ------
    FileNotFoundError
        If the specified configuration file does not exist.
    """
    target_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        content = f.read()

    if yaml is not None:
        config = yaml.safe_load(content)
    else:
        config = _fallback_yaml_parser(content)

    return config or {}


class Config:
    """
    Configuration access wrapper providing programmatic access to configuration values.
    """

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        self._raw_config = load_config(config_path)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a top-level configuration item."""
        return self._raw_config.get(key, default)

    @property
    def raw(self) -> Dict[str, Any]:
        """Access raw configuration dictionary."""
        return self._raw_config
