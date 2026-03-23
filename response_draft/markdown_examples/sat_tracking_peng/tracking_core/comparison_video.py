"""Build side-by-side comparison videos from tracker output clips."""
from __future__ import annotations

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
            if len(fr.shape) == 2:
                fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
            elif fr.shape[2] == 4:
                fr = cv2.cvtColor(fr, cv2.COLOR_BGRA2BGR)
            frames_bgr.append(fr)
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
