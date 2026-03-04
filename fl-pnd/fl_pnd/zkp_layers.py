"""Helpers for Groth16 layer selection/whitelists."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def load_layer_whitelist(config: dict | None) -> tuple[list[str], dict]:
    """Return (prefixes, metadata) from config and optional JSON file."""
    prefixes: list[str] = []
    meta: dict = {}
    if not config:
        return prefixes, meta

    inline = config.get("layer_whitelist")
    if isinstance(inline, Sequence) and not isinstance(inline, (str, bytes)):
        prefixes = [str(p) for p in inline]

    path = config.get("layer_whitelist_path")
    if path:
        file_path = Path(path)
        if file_path.is_file():
            try:
                data = json.loads(file_path.read_text())
            except json.JSONDecodeError:
                print(f"[Groth16] Failed to parse layer whitelist file: {file_path}")
            else:
                prefixes = [str(p) for p in data.get("prefixes", prefixes)]
                if isinstance(data, dict):
                    meta = {k: v for k, v in data.items() if k != "prefixes"}
        else:
            print(f"[Groth16] Layer whitelist file not found: {file_path}")
    meta.setdefault("prefixes", prefixes)
    return prefixes, meta


def flatten_selected_arrays(
    names: Sequence[str],
    arrays: Sequence[np.ndarray],
    prefixes: Iterable[str],
) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
    """Return (flat_vector, matched_names, matched_arrays) for selected prefixes."""
    prefix_list = list(prefixes or [])
    if not prefix_list:
        return np.empty(0, dtype=np.float32), [], []

    selected: list[np.ndarray] = []
    matched: list[str] = []
    for name, arr in zip(names, arrays):
        if any(name.startswith(prefix) for prefix in prefix_list):
            selected.append(np.asarray(arr, dtype=np.float32, order="C"))
            matched.append(name)

    if not selected:
        return np.empty(0, dtype=np.float32), [], []

    if len(selected) == 1:
        flat = selected[0].ravel().astype(np.float32, copy=False)
    else:
        flat = np.concatenate([arr.ravel() for arr in selected]).astype(np.float32, copy=False)
    return flat, matched, selected
