"""
Calibrated noise injection for nadir tracker SNR-sweep experiments.

Noise types
-----------
* **Gaussian readout noise** -- additive N(0, sigma), models sensor electronics.
* **Poisson shot noise** -- signal-dependent, models photon counting statistics.
* **Salt-and-pepper** -- random hot/dead pixels at probability *p*.
* **Background clutter** -- random Gaussian blobs (target-sized) scattered
  across the frame, modelling scene-level false alarms.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── Noise tiers ──────────────────────────────────────────────────────────────
#   Each tier is a dict consumed by ``add_noise``.
#   gaussian_sigma : readout noise std-dev (0-255 scale)
#   poisson_gain   : photon-scaling factor (higher = less noise, 0 = off)
#   snp_prob       : salt-and-pepper probability per pixel
#   clutter_spots  : number of random bright blobs per frame

NOISE_TIERS: dict[str, dict[str, float]] = {
    "clean":       {"gaussian_sigma": 0,  "poisson_gain": 0,    "snp_prob": 0,      "clutter_spots": 0},
    "very_low":    {"gaussian_sigma": 3,  "poisson_gain": 0.8,  "snp_prob": 0,      "clutter_spots": 0},
    "low":         {"gaussian_sigma": 5,  "poisson_gain": 0.5,  "snp_prob": 0.001,  "clutter_spots": 0},
    "moderate":    {"gaussian_sigma": 10, "poisson_gain": 0.3,  "snp_prob": 0.002,  "clutter_spots": 0},
    "medium":      {"gaussian_sigma": 15, "poisson_gain": 0.2,  "snp_prob": 0.003,  "clutter_spots": 2},
    "medium_high": {"gaussian_sigma": 20, "poisson_gain": 0.15, "snp_prob": 0.005,  "clutter_spots": 3},
    "high":        {"gaussian_sigma": 30, "poisson_gain": 0.1,  "snp_prob": 0.005,  "clutter_spots": 5},
    "very_high":   {"gaussian_sigma": 40, "poisson_gain": 0.08, "snp_prob": 0.01,   "clutter_spots": 8},
    "extreme":     {"gaussian_sigma": 50, "poisson_gain": 0.05, "snp_prob": 0.01,   "clutter_spots": 12},
}


# ── Single-frame noise application ──────────────────────────────────────────

def add_noise(
    frame: np.ndarray,
    *,
    gaussian_sigma: float = 0,
    poisson_gain: float = 0,
    snp_prob: float = 0,
    clutter_spots: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply composable noise to a single grayscale frame.

    Parameters
    ----------
    frame : uint8 HxW
    gaussian_sigma : std-dev of additive Gaussian noise (pixel units 0-255).
    poisson_gain : photon-scale factor; frame is scaled to *gain* photons per
        unit intensity before Poisson sampling and scaled back. Higher gain →
        lower relative noise. 0 disables shot noise.
    snp_prob : probability of each pixel being flipped to 0 or 255.
    clutter_spots : number of random Gaussian blobs (radius ~4-8 px,
        intensity 40-120) added to simulate scene false alarms.
    rng : numpy Generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()

    out = frame.astype(np.float64)

    # 1. Poisson shot noise (signal-dependent)
    if poisson_gain > 0:
        scaled = np.maximum(out * poisson_gain, 0)
        out = rng.poisson(scaled).astype(np.float64) / poisson_gain

    # 2. Gaussian readout noise
    if gaussian_sigma > 0:
        out += rng.normal(0, gaussian_sigma, out.shape)

    out = np.clip(out, 0, 255)

    # 3. Salt-and-pepper
    if snp_prob > 0:
        mask = rng.random(out.shape)
        out[mask < snp_prob / 2] = 255.0
        out[mask > 1 - snp_prob / 2] = 0.0

    # 4. Background clutter blobs
    if clutter_spots > 0:
        h, w = out.shape[:2]
        for _ in range(int(clutter_spots)):
            cx = rng.integers(10, w - 10)
            cy = rng.integers(10, h - 10)
            radius = rng.integers(4, 9)
            intensity = rng.integers(40, 120)
            yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            blob = np.exp(-(xx ** 2 + yy ** 2) / (2 * (radius / 2.5) ** 2)) * intensity
            y0, y1 = max(cy - radius, 0), min(cy + radius + 1, h)
            x0, x1 = max(cx - radius, 0), min(cx + radius + 1, w)
            by0, by1 = y0 - (cy - radius), y1 - (cy - radius)
            bx0, bx1 = x0 - (cx - radius), x1 - (cx - radius)
            out[y0:y1, x0:x1] = np.minimum(out[y0:y1, x0:x1] + blob[by0:by1, bx0:bx1], 255)

    return out.astype(np.uint8)


# ── Empirical SNR measurement ───────────────────────────────────────────────

def measure_frame_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Compute empirical SNR in dB between clean and noisy frames."""
    clean_f = clean.astype(np.float64)
    noisy_f = noisy.astype(np.float64)
    signal_power = np.mean(clean_f ** 2)
    noise_power = np.mean((noisy_f - clean_f) ** 2)
    if noise_power < 1e-12:
        return 100.0
    return float(10 * np.log10(signal_power / noise_power))


# ── Noisy video generation ──────────────────────────────────────────────────

def generate_noisy_video(
    input_path: str | Path,
    output_path: str | Path,
    noise_params: dict[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Read *input_path*, apply noise per-frame, write to *output_path*.

    Returns
    -------
    dict with keys: frames, mean_snr_db, output_path
    """
    input_path = str(input_path)
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    snr_samples: list[float] = []
    n_frames = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        gray = bgr[:, :, 0] if bgr.ndim == 3 else bgr
        noisy = add_noise(gray, rng=rng, **noise_params)

        if n_frames % 30 == 0:
            snr_samples.append(measure_frame_snr(gray, noisy))

        bgr_out = cv2.merge([noisy, noisy, noisy])
        out.write(bgr_out)
        n_frames += 1

    cap.release()
    out.release()

    mean_snr = float(np.mean(snr_samples)) if snr_samples else 0.0
    return {
        "frames": n_frames,
        "mean_snr_db": round(mean_snr, 2),
        "output_path": output_path,
    }
