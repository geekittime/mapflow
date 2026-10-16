#!/usr/bin/env python3
"""Validate local model artifacts before starting a MapFlow experiment.

This is intentionally separate from ``audit_source.py``: the source/weight
audit can pass even when a model snapshot is missing or an interrupted
download has left a sparse placeholder.  The model audit follows every shard
listed by ``model.safetensors.index.json`` and opens each shard with
``safetensors`` when that package is available in the reproduction env.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


MODEL_ROOT = Path("/home/wby/models")
SPECS = {
    "Qwen3-1.7B": ("Qwen3-1.7B", 28, 16),
    "Qwen3-8B": ("Qwen3-8B", 36, 32),
    "Qwen3-14B": ("Qwen3-14B", 40, 40),
    "Ministral3-3B": ("Ministral-3-3B-Reasoning-2512", 26, 32),
    "Ministral3-8B": ("Ministral-3-8B-Reasoning-2512", 34, 32),
    "Ministral3-14B": ("Ministral-3-14B-Reasoning-2512", 40, 32),
}


def text_config(config: dict) -> dict:
    return config.get("text_config", config)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_one(label: str, spec: tuple[str, int, int]) -> list[str]:
    dirname, expected_layers, expected_heads = spec
    root = MODEL_ROOT / dirname
    errors: list[str] = []
    if not root.is_dir():
        return [f"{label}: missing directory {root}"]

    config_path = root / "config.json"
    if not config_path.is_file():
        return [f"{label}: missing {config_path}"]
    config = json.loads(config_path.read_text())
    text = text_config(config)
    actual_layers = text.get("num_hidden_layers")
    actual_heads = text.get("num_attention_heads")
    if (actual_layers, actual_heads) != (expected_layers, expected_heads):
        errors.append(
            f"{label}: config layers/heads={(actual_layers, actual_heads)}, "
            f"expected={(expected_layers, expected_heads)}"
        )

    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        missing = [name for name in shard_names if not (root / name).is_file()]
        if missing:
            errors.append(f"{label}: missing shards {missing}")
        partials = sorted(
            str(path.name)
            for path in root.iterdir()
            if path.name.endswith((".part", ".incomplete", ".corrupt"))
        )
        if partials:
            errors.append(f"{label}: partial/corrupt artifacts {partials}")
        try:
            from safetensors import safe_open
        except ImportError:
            safe_open = None
        if safe_open is not None:
            for name in shard_names:
                path = root / name
                if not path.is_file():
                    continue
                try:
                    with safe_open(str(path), framework="pt", device="cpu") as handle:
                        key_count = sum(1 for _ in handle.keys())
                    if key_count == 0:
                        errors.append(f"{label}: empty safetensors shard {name}")
                except Exception as exc:  # noqa: BLE001 - report artifact error
                    errors.append(f"{label}: cannot open {name}: {type(exc).__name__}: {exc}")
        print(
            f"MODEL {label} layers={actual_layers} heads={actual_heads} "
            f"shards={len(shard_names)} config_sha256={sha256(config_path)}"
        )
    else:
        errors.append(f"{label}: missing {index_path}")
    return errors


def main() -> int:
    errors: list[str] = []
    for label, spec in SPECS.items():
        errors.extend(audit_one(label, spec))
    if errors:
        print("MODEL_AUDIT FAIL")
        for error in errors:
            print("ERROR", error)
        return 1
    print("MODEL_AUDIT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
