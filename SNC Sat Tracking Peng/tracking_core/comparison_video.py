"""Build side-by-side comparison videos from tracker output clips."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def _resize_to_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _letterbox_into(frame: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    """Resize frame to fit inside box_w x box_h, preserving aspect; pad with black."""
    frame = _ensure_bgr(frame)
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return np.zeros((box_h, box_w, 3), dtype=np.uint8)
    scale = min(box_w / float(w), box_h / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    x0 = (box_w - nw) // 2
    y0 = (box_h - nh) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _extract_overlay_panel(frame: np.ndarray, reference_width: int) -> np.ndarray:
    """
    Tracker output clips are commonly written as:
    - original | annotated
    - original | binary | annotated

    For the top+grid montage we want only the tracker overlay panel in the grid,
    so crop the rightmost panel when the width suggests a multi-panel export.
    """
    frame = _ensure_bgr(frame)
    h, w = frame.shape[:2]
    if reference_width <= 0 or w <= reference_width:
        return frame

    panel_ratio = w / float(reference_width)
    if panel_ratio >= 2.5:
        panel_w = max(1, int(round(w / 3.0)))
        return frame[:, w - panel_w : w]
    if panel_ratio >= 1.5:
        panel_w = max(1, int(round(w / 2.0)))
        return frame[:, w - panel_w : w]
    return frame


def write_comparison_grid(
    video_paths: list[Path],
    labels: list[str],
    output_path: Path,
    label_height: int = 36,
) -> bool:
    """Stack videos horizontally (same temporal length as shortest). Labels drawn above each column."""
    if not video_paths or len(video_paths) != len(labels):
        return False

    caps = [cv2.VideoCapture(str(p)) for p in video_paths]
    if any(not c.isOpened() for c in caps):
        for c in caps:
            c.release()
        return False

    fps = int(caps[0].get(cv2.CAP_PROP_FPS)) or 25
    widths = [int(c.get(cv2.CAP_PROP_FRAME_WIDTH)) for c in caps]
    heights = [int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)) for c in caps]
    target_h = min(h for h in heights if h > 0) or 240

    col_widths = [max(1, int(round(w * target_h / h))) for w, h in zip(widths, heights)]
    total_w = sum(col_widths)
    canvas_h = target_h + label_height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (total_w, canvas_h))
    if not writer.isOpened():
        for c in caps:
            c.release()
        return False

    for c in caps:
        c.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        frames_bgr: list[np.ndarray] = []
        ok = True
        for c in caps:
            ret, fr = c.read()
            if not ret:
                ok = False
                break
            frames_bgr.append(_ensure_bgr(fr))
        if not ok:
            break

        row = np.zeros((canvas_h, total_w, 3), dtype=np.uint8)
        x = 0
        for fr, cw, lab in zip(frames_bgr, col_widths, labels):
            resized = _resize_to_height(fr, target_h)
            rw = resized.shape[1]
            if rw != cw:
                resized = cv2.resize(resized, (cw, target_h), interpolation=cv2.INTER_AREA)
            row[label_height : label_height + target_h, x : x + cw] = resized
            cv2.putText(
                row,
                lab[:28],
                (x + 4, label_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            x += cw
        writer.write(row)

    writer.release()
    for c in caps:
        c.release()
    return True


def write_comparison_with_original(
    original_video: Path,
    overlay_video_paths: list[Path],
    labels: list[str],
    output_path: Path,
    label_height: int = 36,
) -> bool:
    """Montage: original feed first column, then each tracker overlay video."""
    paths = [original_video] + list(overlay_video_paths)
    labs = ["Input"] + list(labels)
    return write_comparison_grid(paths, labs, output_path, label_height=label_height)


def write_comparison_with_original_top_grid(
    original_video: Path,
    overlay_video_paths: list[Path],
    labels: list[str],
    output_path: Path,
    *,
    max_canvas_width: int = 1440,
    top_label_h: int = 26,
    cell_label_h: int = 20,
    margin: int = 2,
    max_cols: int = 4,
) -> bool:
    """
    Raw input video on top (full canvas width); tracker overlay clips in a tight grid below.

    Each overlay frame is letterboxed into its grid cell. Stops when any capture ends
    (same effective length as the shortest input).
    """
    if not overlay_video_paths or len(overlay_video_paths) != len(labels):
        return False

    cap_orig = cv2.VideoCapture(str(original_video))
    caps = [cv2.VideoCapture(str(p)) for p in overlay_video_paths]
    if not cap_orig.isOpened() or any(not c.isOpened() for c in caps):
        cap_orig.release()
        for c in caps:
            c.release()
        return False

    fps = int(cap_orig.get(cv2.CAP_PROP_FPS)) or int(caps[0].get(cv2.CAP_PROP_FPS)) or 25

    n = len(caps)
    cols = max(1, min(max_cols, int(math.ceil(math.sqrt(n)))))
    rows = int(math.ceil(n / cols))

    W = max(320, int(max_canvas_width))

    ret0, fr0 = cap_orig.read()
    if not ret0 or fr0 is None:
        cap_orig.release()
        for c in caps:
            c.release()
        return False
    fr0 = _ensure_bgr(fr0)
    oh0, ow0 = fr0.shape[:2]
    top_h = max(1, int(round(W * oh0 / max(1, ow0))))

    ret1, fr1 = caps[0].read()
    if not ret1 or fr1 is None:
        cap_orig.release()
        for c in caps:
            c.release()
        return False
    fr1 = _ensure_bgr(fr1)
    vh, vw = fr1.shape[:2]
    cell_aspect = vw / max(1.0, float(vh))

    cell_w = W // cols
    cell_inner_h = max(80, int(round(cell_w / cell_aspect)))
    cell_h = cell_label_h + cell_inner_h + margin

    bottom_h = rows * cell_h + margin
    gap = 6
    canvas_h = top_label_h + top_h + gap + bottom_h
    canvas_w = W

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_w, canvas_h))
    if not writer.isOpened():
        cap_orig.release()
        for c in caps:
            c.release()
        return False

    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for c in caps:
        c.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret_o, frame_o = cap_orig.read()
        if not ret_o or frame_o is None:
            break
        frame_o = _ensure_bgr(frame_o)

        overlay_frames: list[np.ndarray] = []
        ok = True
        for c in caps:
            ret, fr = c.read()
            if not ret or fr is None:
                ok = False
                break
            overlay_frames.append(_extract_overlay_panel(fr, frame_o.shape[1]))
        if not ok:
            break

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        top_resized = cv2.resize(frame_o, (canvas_w, top_h), interpolation=cv2.INTER_AREA)
        cv2.putText(
            canvas,
            "Input (raw)",
            (4, top_label_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        canvas[top_label_h : top_label_h + top_h, 0:canvas_w] = top_resized

        y0 = top_label_h + top_h + gap
        for idx in range(rows * cols):
            r, col = divmod(idx, cols)
            if idx >= n:
                break
            x0 = col * cell_w
            y = y0 + r * cell_h
            lab = labels[idx][:24]
            cv2.putText(
                canvas,
                lab,
                (x0 + margin, y + cell_label_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            inner_y = y + cell_label_h
            inner_w = cell_w - 2 * margin
            inner_h = cell_inner_h
            cell_img = _letterbox_into(overlay_frames[idx], inner_w, inner_h)
            canvas[inner_y : inner_y + inner_h, x0 + margin : x0 + margin + inner_w] = cell_img

        writer.write(canvas)

    writer.release()
    cap_orig.release()
    for c in caps:
        c.release()
    return True
