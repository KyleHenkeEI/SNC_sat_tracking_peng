"""
Air-to-air preprocessing: translation-only scene stabilization (phase correlation)
and optional CLAHE for passive IR contrast.

Used by PoissonMultiBernoulliTracker when a2a_phase_stabilize / a2a_clahe_clip_limit
are enabled. See AIR_TO_AIR_ADAPTATION.md for operational guidance.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def apply_clahe_gray(
    gray: np.ndarray,
    clip_limit: float,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to a single-channel uint8 image. clip_limit 0 or negative disables."""
    if clip_limit <= 0:
        return gray
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def apply_clahe_sequence(
    frames: Sequence[np.ndarray],
    clip_limit: float,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for f in frames:
        g = f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        out.append(apply_clahe_gray(g, clip_limit, tile_grid_size))
    return out


def warp_translate(img: np.ndarray, tx: float, ty: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def warp_translate_color(img: np.ndarray, tx: float, ty: float) -> np.ndarray:
    """Apply the same translation warp to BGR or grayscale video frames."""
    if tx == 0.0 and ty == 0.0:
        return img
    return warp_translate(img, tx, ty)


def stabilize_translation_phase_chain(
    frames: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """
    Align each frame to a common coordinate system by chaining phase-correlation
    shifts between consecutive raw frames. Frame 0 is the reference.

    Assumes mostly global translation (small rotation / parallax will leak).

    Returns aligned grays and per-frame cumulative translation (tx, ty) applied
    to each original gray (same warp should be applied to BGR for overlays).
    """
    if not frames:
        return [], []
    if len(frames) == 1:
        g0 = np.asarray(frames[0], dtype=np.uint8)
        if g0.ndim != 2:
            g0 = cv2.cvtColor(g0, cv2.COLOR_BGR2GRAY)
        return [g0.copy()], [(0.0, 0.0)]

    shifts: list[tuple[float, float]] = [(0.0, 0.0)]
    aligned: list[np.ndarray] = []
    cum_x = 0.0
    cum_y = 0.0
    g0 = np.asarray(frames[0], dtype=np.uint8)
    if g0.ndim != 2:
        g0 = cv2.cvtColor(g0, cv2.COLOR_BGR2GRAY)
    aligned.append(g0.copy())

    for i in range(1, len(frames)):
        prev = np.asarray(frames[i - 1], dtype=np.float32)
        if prev.ndim != 2:
            prev = cv2.cvtColor(prev.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        curr = np.asarray(frames[i], dtype=np.float32)
        if curr.ndim != 2:
            curr = cv2.cvtColor(curr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        try:
            (dx, dy), _resp = cv2.phaseCorrelate(prev, curr)
        except cv2.error:
            dx, dy = 0.0, 0.0
        cum_x += float(dx)
        cum_y += float(dy)
        shifts.append((cum_x, cum_y))
        gi = np.asarray(frames[i], dtype=np.uint8)
        if gi.ndim != 2:
            gi = cv2.cvtColor(gi, cv2.COLOR_BGR2GRAY)
        aligned.append(warp_translate(gi, cum_x, cum_y))

    return aligned, shifts


def preprocess_air_to_air_sequence(
    frames: Sequence[np.ndarray],
    *,
    clahe_clip_limit: float = 0.0,
    clahe_tile_size: int = 8,
    phase_stabilize: bool = False,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """
    Full A2A pre-pipeline on a gray (or BGR) frame list: optional CLAHE, optional
    translation stabilization. Returns uint8 gray frames and per-frame (tx, ty)
    warp applied (identity zeros when phase_stabilize is False).
    """
    gray_list: list[np.ndarray] = []
    for f in frames:
        g = f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gray_list.append(np.asarray(g, dtype=np.uint8))

    if clahe_clip_limit > 0:
        ts = (max(2, int(clahe_tile_size)), max(2, int(clahe_tile_size)))
        gray_list = [apply_clahe_gray(g, clahe_clip_limit, ts) for g in gray_list]

    if phase_stabilize:
        return stabilize_translation_phase_chain(gray_list)

    n = len(gray_list)
    return gray_list, [(0.0, 0.0)] * n
