"""I/O utilities: JSONL, YAML, and directory helpers."""

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries.

    Args:
        path: path to the JSONL file.

    Returns:
        List of parsed dictionaries, one per line.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    """Write a list of dictionaries to a JSONL file.

    Args:
        data: list of dictionaries to serialize.
        path: output file path.
    """
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_yaml(path: str) -> Dict[str, Any]:
    """Read a YAML file into a dictionary.

    Args:
        path: path to the YAML file.

    Returns:
        Parsed dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(data: Dict[str, Any], path: str) -> None:
    """Write a dictionary to a YAML file.

    Args:
        data: dictionary to serialize.
        path: output file path.
    """
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def ensure_dir(path: str) -> None:
    """Create a directory and all parent directories if they do not exist.

    Equivalent to ``mkdir -p``.

    Args:
        path: directory path to create.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
