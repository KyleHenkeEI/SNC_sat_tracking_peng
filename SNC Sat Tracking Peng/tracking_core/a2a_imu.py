"""
Optional IMU-assisted compensation hooks for air-to-air passive tracking.

This module does **not** perform time synchronization or full lever-arm correction.
Use only after camera–IMU extrinsic calibration, time alignment, and validation
on real data. See AIR_TO_AIR_ADAPTATION.md § IMU and latency.

Default behavior is a no-op (zero residual shift) so trackers run unchanged unless
explicitly configured.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics for small-angle gyro → pixel drift approximations."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class IMUSample:
    """One synchronized gyro sample (body or sensor frame per `frame` convention)."""

    frame_index: int
    wx: float
    wy: float
    wz: float
    t_sec: float | None = None


def load_imu_samples_json(path: str | Path) -> list[IMUSample]:
    """
    Load IMU samples from JSON.

    Expected format: list of objects, e.g.
    [{"frame": 0, "wx": 0.0, "wy": 0.0, "wz": 0.0, "t": 0.0}, ...]
    Angular rates in rad/s unless you rescale externally.
    """
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("IMU JSON must be a list of objects")
    out: list[IMUSample] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(
            IMUSample(
                frame_index=int(item["frame"]),
                wx=float(item.get("wx", 0.0)),
                wy=float(item.get("wy", 0.0)),
                wz=float(item.get("wz", 0.0)),
                t_sec=float(item["t"]) if item.get("t") is not None else None,
            )
        )
    out.sort(key=lambda s: s.frame_index)
    return out


def gyro_pixel_drift_small_angle(
    omega_x: float,
    omega_y: float,
    omega_z: float,
    dt: float,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    """
    Crude small-angle model: incremental pixel translation from angular rate
    in the **camera** frame (rad/s), over dt seconds.

    This ignores cross-coupling, distortion, and lever arm; suitable only as a
    stub or for sanity checks after proper projection is implemented.
    """
    # Convention sketch: positive omega_y (pitch rate in camera coords) shifts
    # the image horizontally; tune signs with your calibration.
    dx = float(intrinsics.fx * omega_y * dt)
    dy = float(intrinsics.fy * omega_x * dt)
    _ = omega_z  # in-plane rotation not modeled as pure translation
    return dx, dy


def per_frame_imu_translation_lookup(
    samples: list[IMUSample],
    num_frames: int,
    intrinsics: CameraIntrinsics,
    default_dt: float,
) -> tuple[list[float], list[float]]:
    """
    Build per-frame cumulative (dx, dy) in pixels by integrating gyro samples
    matched by `frame_index`. Missing frames use zero increment.

    Returns parallel lists tx[i], ty[i] to subtract (warp current frame by -t)
    when compensating host rotation in pixel space (prototype only).
    """
    by_frame: dict[int, IMUSample] = {s.frame_index: s for s in samples}
    tx = [0.0] * num_frames
    ty = [0.0] * num_frames
    cum_x = 0.0
    cum_y = 0.0
    for i in range(num_frames):
        s = by_frame.get(i)
        if s is None:
            tx[i] = cum_x
            ty[i] = cum_y
            continue
        dt = default_dt
        if s.t_sec is not None and i > 0:
            prev = by_frame.get(i - 1)
            if prev is not None and prev.t_sec is not None:
                dt = max(1e-6, s.t_sec - prev.t_sec)
        ddx, ddy = gyro_pixel_drift_small_angle(s.wx, s.wy, s.wz, dt, intrinsics)
        cum_x += ddx
        cum_y += ddy
        tx[i] = cum_x
        ty[i] = cum_y
    return tx, ty


def imu_stub_config() -> dict[str, Any]:
    """Metadata for summary.json when IMU path is set but compensation is disabled."""
    return {
        "imu_mode": "stub_or_lookup",
        "note": "Requires calibration and sync; see AIR_TO_AIR_ADAPTATION.md",
    }
