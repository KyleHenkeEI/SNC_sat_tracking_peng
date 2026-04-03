#!/usr/bin/env python3
"""
Textbook-PMB consensus ensemble — full-grid parameter sweep.

Runs every combination of bg_threshold x min_intensity x clutter_rate,
then builds a single consensus video from all runs with spatial voting.
Includes temporal smoothing to reduce jitter in the consensus overlay.

Usage:
    python txtPMB_ensemble.py --input-video path/to/video.mp4
    python txtPMB_ensemble.py --input-video video.mp4 --workers 6 --live-log
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from run_ensemble import (
    _ensemble_tracker_job,
    _PrefixedStream,
    _voter_color_bgr,
    _ensure_bgr,
    build_per_frame_consensus,
    link_consensus_across_frames,
    parse_pmb_track_entries,
    run_single_ensemble_tracker,
    write_ensemble_montage,
)


def _safe_live_job(job: dict[str, Any]) -> dict[str, Any]:
    """Thread-safe live-log runner — always wraps the real fd, not sys.stdout."""
    prefix = job["ensemble_key"]
    real_out = sys.__stdout__
    real_err = sys.__stderr__
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _PrefixedStream(prefix, real_out)
    sys.stderr = _PrefixedStream(prefix, real_err)
    try:
        return run_single_ensemble_tracker({**job, "quiet": False})
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = old_out
        sys.stderr = old_err


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

SWEEP_BG_THRESHOLD = [10, 13, 16, 18, 20]
SWEEP_MIN_INTENSITY = [10, 16, 22, 28, 34]
SWEEP_CLUTTER_RATE = [1.5, 3.5, 6.0, 10.0, 15.0]

SHARED_OVERRIDES: dict[str, Any] = {
    "existence_threshold": 0.50,
    "min_display_confidence": 0.48,
    "detection_prob": 0.75,
    "min_track_length": 5,
    "textbook_mahalanobis_gate_sq": 9.21,
    "max_speed": 120,
    "max_acceleration": 15,
    "max_direction_change": 20,
}


def build_full_grid() -> list[dict[str, Any]]:
    """All combinations of the three sweep axes."""
    runs: list[dict[str, Any]] = []
    for bg, mi, cr in itertools.product(SWEEP_BG_THRESHOLD, SWEEP_MIN_INTENSITY, SWEEP_CLUTTER_RATE):
        key = f"bg{bg}_mi{mi}_cr{cr:.1f}"
        label = f"bg={bg} mi={mi} cr={cr}"
        overrides = {
            **SHARED_OVERRIDES,
            "bg_threshold": bg,
            "min_intensity": mi,
            "clutter_rate": cr,
        }
        runs.append({
            "key": key,
            "tracker_name": "Textbook PMB",
            "label": label,
            "overrides": overrides,
        })
    return runs


# ---------------------------------------------------------------------------
# Temporal smoothing for consensus tracks (reduces jitter)
# ---------------------------------------------------------------------------

def smooth_consensus_tracks(
    tracks: list[dict[str, Any]],
    alpha: float = 0.45,
) -> list[dict[str, Any]]:
    """Apply exponential moving average to consensus track positions.

    alpha controls responsiveness: 0 = fully smooth (lag), 1 = no smoothing.
    0.45 balances jitter reduction with position accuracy.
    """
    smoothed: list[dict[str, Any]] = []
    for tr in tracks:
        hist = tr["history"]
        if len(hist) < 2:
            smoothed.append(tr)
            continue
        new_hist = [hist[0]]
        sx, sy = float(hist[0][1]), float(hist[0][2])
        for i in range(1, len(hist)):
            f, x, y, v = hist[i]
            sx = alpha * x + (1.0 - alpha) * sx
            sy = alpha * y + (1.0 - alpha) * sy
            new_hist.append((f, sx, sy, v))
        smoothed.append({"id": tr["id"], "history": new_hist})
    return smoothed


# ---------------------------------------------------------------------------
# Consensus renderer — always shows marker at last known position
# ---------------------------------------------------------------------------

def render_consensus_video_smooth(
    input_video: Path,
    output_path: Path,
    linked_tracks: list[dict[str, Any]],
    n_trackers: int,
    *,
    max_tail: int = 15,
    fade_frames: int = 6,
) -> bool:
    """Render consensus overlay with persistent markers during track gaps.

    Unlike the default renderer, the bounding box stays visible at the last
    known position for up to ``max_tail`` frames after the last consensus vote,
    fading in opacity so ghost tracks are visually distinct from active ones.
    """
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        return False
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not out.isOpened():
        cap.release()
        return False

    min_consensus_len = 3
    frame_idx = 0

    while True:
        ret, fr = cap.read()
        if not ret or fr is None:
            break
        canvas = _ensure_bgr(fr).copy()

        for tr in linked_tracks:
            if len(tr["history"]) < min_consensus_len:
                continue
            hist = tr["history"]

            tail: list[tuple[int, int, int]] = []
            last_frame_seen = -1
            for ff, x, y, v in hist:
                if ff > frame_idx:
                    break
                ix, iy = int(round(x)), int(round(y))
                tail.append((ix, iy, v))
                last_frame_seen = ff

            gap = frame_idx - last_frame_seen
            if gap > max_tail:
                continue

            if max_tail > 0 and len(tail) > max_tail:
                tail = tail[-max_tail:]

            if len(tail) > 1:
                for i in range(len(tail) - 1):
                    _, _, v1 = tail[i + 1]
                    color = _voter_color_bgr(v1, n_trackers)
                    cv2.line(canvas, tail[i][:2], tail[i + 1][:2], color, 1, cv2.LINE_AA)

            if tail:
                hx, hy, hv = tail[-1]
                color = _voter_color_bgr(hv, n_trackers)
                if gap > 0 and fade_frames > 0:
                    fade = max(0.2, 1.0 - gap / float(fade_frames))
                    color = (
                        int(color[0] * fade),
                        int(color[1] * fade),
                        int(color[2] * fade),
                    )
                cv2.circle(canvas, (hx, hy), 2, color, -1, cv2.LINE_AA)
                cv2.rectangle(canvas, (hx - 3, hy - 3), (hx + 3, hy + 3), color, 1, cv2.LINE_AA)

        out.write(canvas)
        frame_idx += 1

    out.release()
    cap.release()
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Textbook-PMB full-grid consensus — all combinations of "
            "bg_threshold x min_intensity x clutter_rate."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input-video", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--quorum", type=int, default=None,
                   help="Min runs agreeing (default: ceil(n_runs / 3))")
    p.add_argument("--spatial-threshold", type=float, default=6.0)
    p.add_argument("--link-threshold", type=float, default=None)
    p.add_argument("--max-frame-gap", type=int, default=8,
                   help="Frames a consensus track can survive without a vote (default 8, reduces jitter)")
    p.add_argument("--smooth-alpha", type=float, default=0.45,
                   help="EMA smoothing factor for consensus positions (0=max smooth, 1=none, default 0.45)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--live-log", action="store_true")
    p.add_argument("--status-interval", type=float, default=15.0)
    p.add_argument("--preset", type=str, default=None)
    p.add_argument("--preset-file", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    workspace = Path(__file__).resolve().parent
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))

    input_video = args.input_video.resolve()
    if not input_video.is_file():
        print(f"Input video not found: {input_video}", file=sys.stderr)
        return 1

    runs = build_full_grid()
    n_runs = len(runs)

    spatial_threshold = args.spatial_threshold
    link_thr = args.link_threshold if args.link_threshold is not None else spatial_threshold * 2.0
    quorum = args.quorum if args.quorum is not None else max(2, math.ceil(n_runs / 3))
    max_frame_gap = args.max_frame_gap
    smooth_alpha = args.smooth_alpha

    run_dir = args.output_dir
    if run_dir is None:
        run_dir = workspace / "_txtpmb_fullgrid" / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    preset_file_str = str(args.preset_file.resolve()) if args.preset_file else None

    print(f"Textbook-PMB full-grid sweep", flush=True)
    print(f"  bg_threshold   : {SWEEP_BG_THRESHOLD}", flush=True)
    print(f"  min_intensity  : {SWEEP_MIN_INTENSITY}", flush=True)
    print(f"  clutter_rate   : {SWEEP_CLUTTER_RATE}", flush=True)
    print(f"  Combinations   : {len(SWEEP_BG_THRESHOLD)} x {len(SWEEP_MIN_INTENSITY)} x {len(SWEEP_CLUTTER_RATE)} = {n_runs} runs", flush=True)
    print(f"  Quorum         : {quorum}/{n_runs}", flush=True)
    print(f"  Fusion         : spatial={spatial_threshold}px, link={link_thr}px, max_gap={max_frame_gap}f", flush=True)
    print(f"  Smoothing      : alpha={smooth_alpha}", flush=True)
    print(f"  Output         : {run_dir}", flush=True)

    # ---- Build job payloads ----
    jobs: list[dict[str, Any]] = []
    for cfg in runs:
        sub = run_dir / cfg["key"]
        jobs.append({
            "workspace_root": str(workspace),
            "ensemble_key": cfg["key"],
            "tracker_name": cfg["tracker_name"],
            "input_video": str(input_video),
            "output_dir": str(sub),
            "overrides": cfg["overrides"],
            "preset_name": args.preset,
            "preset_file": preset_file_str,
            "quiet": not args.live_log,
        })

    # ---- Save resolved config ----
    resolved = {
        "sweep_bg_threshold": SWEEP_BG_THRESHOLD,
        "sweep_min_intensity": SWEEP_MIN_INTENSITY,
        "sweep_clutter_rate": SWEEP_CLUTTER_RATE,
        "n_runs": n_runs,
        "quorum": quorum,
        "fusion": {
            "spatial_threshold_px": spatial_threshold,
            "link_threshold_px": link_thr,
            "max_frame_gap": max_frame_gap,
            "smooth_alpha": smooth_alpha,
        },
        "runs": [{"key": r["key"], "overrides": r["overrides"]} for r in runs],
    }
    (run_dir / "txtpmb_config_resolved.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8",
    )

    # ---- Run all trackers in parallel ----
    results: list[dict[str, Any]] = []
    future_to_key: dict[Any, str] = {}
    stop_status = threading.Event()

    def status_loop() -> None:
        while True:
            if stop_status.wait(timeout=args.status_interval):
                break
            pending = [future_to_key[f] for f in list(future_to_key) if not f.done()]
            done = n_runs - len(pending)
            if pending:
                print(
                    f"[txtPMB] Progress: {done}/{n_runs} complete, "
                    f"{len(pending)} running",
                    flush=True,
                )

    status_thread: threading.Thread | None = None
    if args.status_interval > 0:
        status_thread = threading.Thread(target=status_loop, daemon=True)

    mode_label = "live-log (threads)" if args.live_log else f"process pool ({args.workers} workers)"
    print(f"\nRunning {n_runs} trackers — {mode_label}\n", flush=True)

    try:
        if args.live_log:
            n_workers = min(args.workers, len(jobs))
            with ThreadPoolExecutor(max_workers=max(1, n_workers)) as ex:
                for j in jobs:
                    fut = ex.submit(_safe_live_job, j)
                    future_to_key[fut] = j["ensemble_key"]
                if status_thread is not None:
                    status_thread.start()
                for fut in as_completed(future_to_key):
                    key = future_to_key[fut]
                    try:
                        summ = fut.result()
                        results.append(summ)
                        done = len(results)
                        print(f"[txtPMB] ({done}/{n_runs}) Finished {key} -> {summ.get('status', '?')}", flush=True)
                    except Exception as exc:
                        print(f"[txtPMB] {key} -> exception: {exc}", file=sys.stderr, flush=True)
                        results.append({"ensemble_key": key, "status": "error", "error": str(exc)})
        else:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
                for j in jobs:
                    fut = ex.submit(_ensemble_tracker_job, j)
                    future_to_key[fut] = j["ensemble_key"]
                if status_thread is not None:
                    status_thread.start()
                for fut in as_completed(future_to_key):
                    key = future_to_key[fut]
                    try:
                        summ = fut.result()
                        results.append(summ)
                        done = len(results)
                        print(f"  ({done}/{n_runs}) [{key}] -> {summ.get('status', '?')}", flush=True)
                    except Exception as exc:
                        print(f"  [{key}] -> exception: {exc}", file=sys.stderr, flush=True)
                        results.append({"ensemble_key": key, "status": "error", "error": str(exc)})
    finally:
        stop_status.set()
        if status_thread is not None and status_thread.is_alive():
            status_thread.join(timeout=2.0)

    failed = [r for r in results if r.get("status") != "ok"]
    if failed:
        print(f"\nWarning: {len(failed)}/{n_runs} runs failed.", file=sys.stderr, flush=True)

    # ---- Collect all track logs ----
    all_entries: list[tuple[int, list[tuple[int, int, float, float]]]] = []
    for cfg in runs:
        sub = run_dir / cfg["key"]
        tlog = sub / "tracks.txt"
        if tlog.is_file():
            tidx = len(all_entries)
            all_entries.append((tidx, parse_pmb_track_entries(tlog)))

    n_ok = len(all_entries)
    if n_ok == 0:
        print("No successful runs; cannot build consensus.", file=sys.stderr)
        return 1

    effective_quorum = min(quorum, n_ok)
    print(f"\n=== Building consensus from {n_ok} track logs (quorum={effective_quorum}) ===", flush=True)

    renumbered = [(i, e) for i, (_s, e) in enumerate(all_entries)]
    consensus_by_frame = build_per_frame_consensus(
        renumbered,
        spatial_threshold=spatial_threshold,
        quorum=effective_quorum,
        n_trackers=n_ok,
    )
    linked = link_consensus_across_frames(
        consensus_by_frame,
        link_threshold=link_thr,
        max_frame_gap=max_frame_gap,
    )

    if smooth_alpha < 1.0:
        linked = smooth_consensus_tracks(linked, alpha=smooth_alpha)

    print(f"  {len(consensus_by_frame)} frames with votes, {len(linked)} consensus tracks", flush=True)

    # ---- Render consensus overlay ----
    consensus_path = run_dir / "consensus_fullgrid.mp4"
    print("[txtPMB] Rendering consensus overlay...", flush=True)
    if not render_consensus_video_smooth(
        input_video, consensus_path, linked, n_ok,
        max_tail=15, fade_frames=6,
    ):
        print("Failed to write consensus video.", file=sys.stderr)
        return 1

    # ---- Save manifest ----
    manifest = {
        "input_video": str(input_video),
        "output_dir": str(run_dir),
        "sweep_bg_threshold": SWEEP_BG_THRESHOLD,
        "sweep_min_intensity": SWEEP_MIN_INTENSITY,
        "sweep_clutter_rate": SWEEP_CLUTTER_RATE,
        "n_runs": n_runs,
        "n_ok": n_ok,
        "quorum": effective_quorum,
        "spatial_threshold_px": spatial_threshold,
        "link_threshold_px": link_thr,
        "max_frame_gap": max_frame_gap,
        "smooth_alpha": smooth_alpha,
        "consensus_frames": len(consensus_by_frame),
        "consensus_tracks": len(linked),
        "tracker_results": results,
        "consensus_video": str(consensus_path),
    }
    (run_dir / "txtpmb_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print("Done.", flush=True)
    print(f"  Consensus video : {consensus_path}", flush=True)
    print(f"  Manifest        : {run_dir / 'txtpmb_manifest.json'}", flush=True)
    print(f"  Config          : {run_dir / 'txtpmb_config_resolved.json'}", flush=True)
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
