"""Shared tracker preset loading and merge helpers."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_PRESET_FILE = Path(__file__).resolve().parent.parent / "tracker_presets.json"


def get_preset_file_path(preset_file: str | Path | None = None) -> Path:
    if preset_file is None:
        return DEFAULT_PRESET_FILE
    return Path(preset_file).resolve()


def load_preset_catalog(preset_file: str | Path | None = None) -> dict[str, Any]:
    path = get_preset_file_path(preset_file)
    if not path.is_file():
        raise FileNotFoundError(f"Preset file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Preset file must contain a JSON object: {path}")
    return data


def preset_descriptions(preset_file: str | Path | None = None) -> dict[str, str]:
    data = load_preset_catalog(preset_file)
    out: dict[str, str] = {}
    for name, spec in data.items():
        if not isinstance(spec, dict):
            continue
        out[name] = str(spec.get("description", "")).strip()
    return out


def resolve_preset(
    preset_name: str | None,
    preset_file: str | Path | None = None,
) -> tuple[str | None, dict[str, Any], Path]:
    path = get_preset_file_path(preset_file)
    if not preset_name:
        return None, {}, path

    catalog = load_preset_catalog(path)
    if preset_name not in catalog:
        known = ", ".join(sorted(catalog))
        raise KeyError(f"Unknown preset '{preset_name}'. Available presets: {known}")

    preset = catalog[preset_name]
    if not isinstance(preset, dict):
        raise ValueError(f"Preset '{preset_name}' must be a JSON object in {path}")
    return preset_name, deepcopy(preset), path


def apply_preset_to_kwargs(
    kwargs: dict[str, Any],
    spec: dict[str, Any],
    preset_name: str | None = None,
    preset_file: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preset_name, preset, path = resolve_preset(preset_name, preset_file)
    merged = deepcopy(kwargs)
    if preset:
        merged.update(preset.get("global", {}))
        merged.update(preset.get("kinds", {}).get(spec["kind"], {}))
        merged.update(preset.get("families", {}).get(spec["family"], {}))
        merged.update(preset.get("trackers", {}).get(spec["name"], {}))
    meta = {
        "preset_name": preset_name,
        "preset_file": str(path),
    }
    return merged, meta
