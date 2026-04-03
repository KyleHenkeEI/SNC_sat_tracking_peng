"""
Ego-motion compensation for nadir (space-looking-down) tracking.

Provides homography/affine registration between consecutive frames so that
the dominant Earth-background motion is removed.  Anything that does NOT
move with the background (missiles, aircraft, …) appears as a residual.
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Feature-based registration
# ---------------------------------------------------------------------------

def estimate_homography_orb(
    prev: np.ndarray,
    curr: np.ndarray,
    max_features: int = 1000,
    good_ratio: float = 0.75,
    ransac_reproj: float = 3.0,
) -> tuple[np.ndarray | None, int]:
    """Estimate a homography *prev → curr* using ORB + brute-force matching.

    Returns (H, n_inliers).  H is None when registration fails.
    """
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(prev, None)
    kp2, des2 = orb.detectAndCompute(curr, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < good_ratio * n.distance:
                good.append(m)

    if len(good) < 4:
        return None, 0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_reproj)
    if H is None:
        return None, 0

    n_inliers = int(mask.ravel().sum()) if mask is not None else 0
    return H, n_inliers


def estimate_affine_orb(
    prev: np.ndarray,
    curr: np.ndarray,
    max_features: int = 1000,
    good_ratio: float = 0.75,
    ransac_reproj: float = 3.0,
) -> tuple[np.ndarray | None, int]:
    """Estimate a rigid (partial-affine) transform *prev → curr*.

    Returns (M_2x3, n_inliers).
    """
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(prev, None)
    kp2, des2 = orb.detectAndCompute(curr, None)

    if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < good_ratio * n.distance:
                good.append(m)

    if len(good) < 3:
        return None, 0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC,
                                          ransacReprojThreshold=ransac_reproj)
    if M is None:
        return None, 0

    n_inliers = int(mask.ravel().sum()) if mask is not None else 0
    return M, n_inliers


# ---------------------------------------------------------------------------
# Phase-correlation fallback (fast, translation-only)
# ---------------------------------------------------------------------------

def estimate_translation_phase(
    prev: np.ndarray,
    curr: np.ndarray,
) -> tuple[float, float]:
    """Return (dx, dy) translation from phase correlation."""
    pf = prev.astype(np.float32)
    cf = curr.astype(np.float32)
    try:
        (dx, dy), _ = cv2.phaseCorrelate(pf, cf)
    except cv2.error:
        dx, dy = 0.0, 0.0
    return float(dx), float(dy)


# ---------------------------------------------------------------------------
# Motion-compensated residual
# ---------------------------------------------------------------------------

def warp_frame(frame: np.ndarray, H: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Warp *frame* with 3x3 homography or 2x3 affine, output shape (w, h)."""
    if H.shape[0] == 2:
        return cv2.warpAffine(frame, H, size,
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return cv2.warpPerspective(frame, H, size,
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def motion_compensated_residual(
    prev: np.ndarray,
    curr: np.ndarray,
    method: str = "homography",
    max_features: int = 1000,
    good_ratio: float = 0.75,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Warp *prev* into *curr*'s coordinate system and return the residual.

    Parameters
    ----------
    method : ``"homography"`` | ``"affine"`` | ``"phase"``

    Returns
    -------
    residual : uint8 absolute difference image
    H        : transform matrix (None for phase)
    n_inliers: number of RANSAC inliers (0 for phase)
    """
    h, w = curr.shape[:2]

    if method == "homography":
        H, n_inliers = estimate_homography_orb(prev, curr, max_features, good_ratio)
        if H is None:
            return cv2.absdiff(prev, curr), None, 0
        warped = warp_frame(prev, H, (w, h))
        return cv2.absdiff(warped, curr), H, n_inliers

    if method == "affine":
        M, n_inliers = estimate_affine_orb(prev, curr, max_features, good_ratio)
        if M is None:
            return cv2.absdiff(prev, curr), None, 0
        warped = warp_frame(prev, M, (w, h))
        return cv2.absdiff(warped, curr), M, n_inliers

    # phase-correlation fallback
    dx, dy = estimate_translation_phase(prev, curr)
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    warped = cv2.warpAffine(prev, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    return cv2.absdiff(warped, curr), M, 0


# ---------------------------------------------------------------------------
# Multi-frame residual accumulation
# ---------------------------------------------------------------------------

def multi_frame_residual(
    frames: list[np.ndarray],
    ref_idx: int = -1,
    method: str = "homography",
) -> np.ndarray:
    """Register all *frames* to ``frames[ref_idx]`` and return a fused residual.

    The fused image is ``0.6 * max + 0.4 * mean`` over all pair-wise residuals,
    matching the TBD-style accumulation already used in the sat-tracking code.
    """
    ref = frames[ref_idx]
    residuals = []
    for i, f in enumerate(frames):
        if i == (ref_idx % len(frames)):
            continue
        r, _, _ = motion_compensated_residual(f, ref, method=method)
        residuals.append(r.astype(np.float32))

    if not residuals:
        return np.zeros_like(ref)

    stacked = np.stack(residuals, axis=0)
    fused = 0.6 * np.max(stacked, axis=0) + 0.4 * np.mean(stacked, axis=0)
    return np.clip(fused, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Adaptive thresholding for structured backgrounds
# ---------------------------------------------------------------------------

def adaptive_local_threshold(
    residual: np.ndarray,
    block_size: int = 31,
    C: float = 5.0,
    global_floor: int = 8,
) -> np.ndarray:
    """Threshold a residual image using local adaptive + global floor.

    Pixels must exceed *both* the local adaptive threshold and ``global_floor``
    to be flagged as foreground.
    """
    blurred = cv2.GaussianBlur(residual, (3, 3), 0.7)
    bs = block_size if block_size % 2 == 1 else block_size + 1
    local = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, bs, -C)
    _, hard = cv2.threshold(blurred, global_floor, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(local, hard)


def sigma_clip_threshold(
    residual: np.ndarray,
    sigma: float = 4.0,
    min_threshold: int = 6,
) -> np.ndarray:
    """Threshold based on local sigma-clipping: pixel > mean + sigma*std."""
    blurred = cv2.GaussianBlur(residual, (3, 3), 0.7)
    mean = cv2.GaussianBlur(blurred.astype(np.float32), (31, 31), 0)
    sq_mean = cv2.GaussianBlur((blurred.astype(np.float32)) ** 2, (31, 31), 0)
    std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))
    threshold_map = (mean + sigma * std).astype(np.float32)
    threshold_map = np.maximum(threshold_map, float(min_threshold))
    binary = (blurred.astype(np.float32) > threshold_map).astype(np.uint8) * 255
    return binary
