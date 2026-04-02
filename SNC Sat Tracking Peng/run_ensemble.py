#!/usr/bin/env python3
"""
Run multiple PMB / PMBM variants in parallel, fuse tracks by spatial voting, then write
consensus_overlay.mp4 and ensemble_montage.mp4.

Parameter sets: edit ensemble_config.json next to this script (or pass --ensemble-config).
Progress: use --live-log for prefixed per-tracker frame updates; optional --status-interval
prints which runs are still active (works with process pool too).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Single-tracker run (used by ProcessPool, ThreadPool + live log, and tests)
# -----------------------------------------------------------------------------


def run_single_ensemble_tracker(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one tracker variant; write summary.json like tracker_cli."""
    root = Path(payload["workspace_root"])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import json as _json
    import time as _time

    from tracking_core.variants import TRACKER_VARIANTS, instantiate_tracker

    tracker_name = payload["tracker_name"]
    spec = None
    for s in TRACKER_VARIANTS:
        if s["name"] == tracker_name:
            spec = deepcopy(s)
            break
    if spec is None:
        raise ValueError(f"Unknown tracker: {tracker_name}")

    base_ov = dict(spec.get("overrides") or {})
    base_ov.update(payload["overrides"])
    spec["overrides"] = base_ov

    input_video = str(Path(payload["input_video"]).resolve())
    output_dir = Path(payload["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "tracker_name": tracker_name,
        "ensemble_key": payload["ensemble_key"],
        "output_tag": spec["output_tag"],
        "input_video": input_video,
        "output_dir": str(output_dir),
        "status": "pending",
        "overrides_applied": payload["overrides"],
    }

    try:
        tracker = instantiate_tracker(
            spec,
            input_video,
            output_dir,
            preset_name=payload.get("preset_name"),
            preset_file=payload.get("preset_file"),
        )
        if payload.get("quiet", True):
            tracker.verbose = False

        summary["output_video"] = str(Path(tracker.output_video_path).resolve())
        summary["track_log"] = str(Path(tracker.track_log_path).resolve())
        summary["write_video_output"] = bool(getattr(tracker, "write_video_output", True))

        t0 = _time.perf_counter()
        result = tracker.run()
        wall_s = _time.perf_counter() - t0

        summary["wall_runtime_s"] = round(wall_s, 2)
        tr = result.get("runtime_s")
        summary["runtime_s"] = round(float(tr), 2) if tr is not None else round(wall_s, 2)
        ve = result.get("video_encode_runtime_s")
        summary["video_encode_runtime_s"] = round(float(ve), 2) if ve is not None else None
        summary["status"] = "ok"
        summary["frames_processed"] = result.get("frames_processed")
        summary["total_detections"] = result.get("total_detections")
        summary["tracks_metric"] = result.get("unique_tracks", result.get("log_entries"))
        if result.get("air_to_air"):
            summary["air_to_air"] = result["air_to_air"]
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "error"
        summary["error"] = str(exc)

    (output_dir / "summary.json").write_text(_json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _ensemble_tracker_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Multiprocessing entry point (must be top-level for pickle on Windows)."""
    return run_single_ensemble_tracker(payload)


class _PrefixedStream:
    """Line-buffer stdout/stderr with a [key] prefix (thread-safe)."""

    _global_lock = threading.Lock()

    def __init__(self, prefix: str, underlying: TextIO):
        self.prefix = prefix
        self._u = underlying
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        with self._global_lock:
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self._u.write(f"[{self.prefix}] {line}\n")
            self._u.flush()
        return len(s)

    def flush(self) -> None:
        with self._global_lock:
            if self._buf:
                self._u.write(f"[{self.prefix}] {self._buf}")
                self._buf = ""
            self._u.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._u, name)


def _run_live_monitored_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run in a thread with prefixed verbose tracker logs."""
    prefix = job["ensemble_key"]
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _PrefixedStream(prefix, old_out)
    sys.stderr = _PrefixedStream(prefix, old_err)
    j = {**job, "quiet": False}
    try:
        return run_single_ensemble_tracker(j)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = old_out
        sys.stderr = old_err


# -----------------------------------------------------------------------------
# Default ensemble definitions (plan)
# -----------------------------------------------------------------------------

DEFAULT_ENSEMBLE_RUNS: list[dict[str, Any]] = [
    {
        "key": "pmb_conservative",
        "tracker_name": "Current PMB",
        "label": "PMB cons.",
        "overrides": {
            "existence_threshold": 0.60,
            "min_display_confidence": 0.82,
            "clutter_rate": 3.5,
            "min_track_length": 6,
        },
    },
    {
        "key": "pmb_permissive",
        "tracker_name": "Current PMB",
        "label": "PMB perm.",
        "overrides": {
            "existence_threshold": 0.42,
            "min_display_confidence": 0.65,
            "clutter_rate": 7.0,
            "min_track_length": 4,
        },
    },
    {
        "key": "pmbm_conservative",
        "tracker_name": "PMBM",
        "label": "PMBM cons.",
        "overrides": {
            "pmbm_k_best": 3,
            "pmbm_max_hypotheses": 6,
            "existence_threshold": 0.58,
            "min_display_confidence": 0.55,
        },
    },
    {
        "key": "pmbm_permissive",
        "tracker_name": "PMBM",
        "label": "PMBM perm.",
        "overrides": {
            "pmbm_k_best": 8,
            "pmbm_max_hypotheses": 16,
            "existence_threshold": 0.40,
            "min_display_confidence": 0.40,
        },
    },
]


def default_ensemble_config_document() -> dict[str, Any]:
    """Template for ensemble_config.json (editable parameter sets)."""
    return {
        "description": (
            "Edit each run's 'overrides' (merged onto the tracker registry defaults). "
            "tracker_name must match TRACKER_VARIANTS, e.g. 'Current PMB' or 'PMBM'. "
            "Optional 'fusion' overrides quorum / voting distances when those keys are present."
        ),
        "fusion": {
            "quorum": 2,
            "spatial_threshold_px": 8.0,
            "link_threshold_px": None,
        },
        "runs": deepcopy(DEFAULT_ENSEMBLE_RUNS),
    }


def load_ensemble_config_file(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Load runs + optional fusion overrides from JSON.
    Returns (runs, fusion_dict). fusion may be empty.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    fusion = raw.get("fusion") or {}
    if not isinstance(fusion, dict):
        raise ValueError(f"'fusion' must be an object if present: {path}")
    if "pool_overrides" in fusion and not isinstance(fusion["pool_overrides"], dict):
        raise ValueError(f"'fusion.pool_overrides' must be an object if present: {path}")
    runs = raw.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"'runs' must be a non-empty array: {path}")
    out_runs: list[dict[str, Any]] = []
    for i, r in enumerate(runs):
        if not isinstance(r, dict):
            raise ValueError(f"runs[{i}] must be an object: {path}")
        for k in ("key", "tracker_name", "label", "overrides"):
            if k not in r:
                raise ValueError(f"runs[{i}] missing required key '{k}': {path}")
        if not isinstance(r["overrides"], dict):
            raise ValueError(f"runs[{i}].overrides must be an object: {path}")
        entry = {
            "key": str(r["key"]),
            "tracker_name": str(r["tracker_name"]),
            "label": str(r["label"]),
            "overrides": dict(r["overrides"]),
        }
        if "pool" in r:
            entry["pool"] = str(r["pool"])
        out_runs.append(entry)
    return out_runs, fusion


def resolve_ensemble_runs(
    workspace: Path,
    *,
    ensemble_config: Path | None,
    builtin_runs: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path | None]:
    """
    Choose runs list and fusion snippet. Returns (runs, fusion, config_path_or_none).
    """
    if builtin_runs:
        return deepcopy(DEFAULT_ENSEMBLE_RUNS), {}, None
    if ensemble_config is not None:
        if not ensemble_config.is_file():
            raise FileNotFoundError(f"Ensemble config not found: {ensemble_config}")
        runs, fusion = load_ensemble_config_file(ensemble_config.resolve())
        return runs, fusion, ensemble_config.resolve()
    auto = workspace / "ensemble_config.json"
    if auto.is_file():
        runs, fusion = load_ensemble_config_file(auto.resolve())
        return runs, fusion, auto.resolve()
    return deepcopy(DEFAULT_ENSEMBLE_RUNS), {}, None


# -----------------------------------------------------------------------------
# Track log parsing & consensus
# -----------------------------------------------------------------------------


def parse_pmb_track_entries(path: Path) -> list[tuple[int, int, float, float]]:
    """Return list of (frame, track_id, x, y) from a PMB-family tracks.txt."""
    rows: list[tuple[int, int, float, float]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            frame = int(parts[0])
            tid = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            rows.append((frame, tid, x, y))
        except ValueError:
            continue
    return rows


def _union_find_cluster(
    points: list[tuple[int, float, float]],
    threshold_px: float,
) -> list[list[int]]:
    """Cluster indices; points[i] = (tracker_idx, x, y)."""
    n = len(points)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    thr2 = threshold_px * threshold_px
    for i in range(n):
        _, xi, yi = points[i]
        for j in range(i + 1, n):
            _, xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            if dx * dx + dy * dy <= thr2:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def build_per_frame_consensus(
    tracker_entries: list[tuple[int, list[tuple[int, int, float, float]]]],
    *,
    spatial_threshold: float,
    quorum: int,
    n_trackers: int | None = None,
) -> dict[int, list[tuple[float, float, int]]]:
    """
    tracker_entries: (tracker_idx, list of (frame, tid, x, y))
    Returns frame -> list of (x, y, voter_count) consensus detections.
    """
    by_frame: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for tidx, entries in tracker_entries:
        for frame, _tid, x, y in entries:
            by_frame[frame].append((tidx, float(x), float(y)))

    consensus: dict[int, list[tuple[float, float, int]]] = {}
    for frame, pts in sorted(by_frame.items()):
        if not pts:
            continue
        clusters = _union_find_cluster(pts, spatial_threshold)
        out: list[tuple[float, float, int]] = []
        for idxs in clusters:
            trackers_here = {pts[i][0] for i in idxs}
            if len(trackers_here) < quorum:
                continue
            xs = [pts[i][1] for i in idxs]
            ys = [pts[i][2] for i in idxs]
            out.append((float(np.mean(xs)), float(np.mean(ys)), len(trackers_here)))
        if out:
            consensus[frame] = out
    return consensus


def link_consensus_across_frames(
    consensus_by_frame: dict[int, list[tuple[float, float, int]]],
    *,
    link_threshold: float,
    max_frame_gap: int = 4,
) -> list[dict[str, Any]]:
    """Greedy nearest-neighbor linking; each track is {id, history: [(f,x,y,voters), ...]}."""
    sorted_frames = sorted(consensus_by_frame.keys())
    tracks: list[dict[str, Any]] = []
    next_id = 1

    for f in sorted_frames:
        points = consensus_by_frame[f]
        pairs: list[tuple[float, int, int]] = []

        for pi, (x, y, _v) in enumerate(points):
            for j, tr in enumerate(tracks):
                lf, lx, ly, _lv = tr["history"][-1]
                if f - lf > max_frame_gap:
                    continue
                d = math.hypot(x - lx, y - ly)
                if d <= link_threshold:
                    pairs.append((d, pi, j))

        pairs.sort(key=lambda t: t[0])
        used_p: set[int] = set()
        used_t: set[int] = set()
        for _d, pi, j in pairs:
            if pi in used_p or j in used_t:
                continue
            x, y, v = points[pi]
            tracks[j]["history"].append((f, x, y, v))
            used_p.add(pi)
            used_t.add(j)

        for pi, (x, y, v) in enumerate(points):
            if pi not in used_p:
                tracks.append({"id": next_id, "history": [(f, x, y, v)]})
                next_id += 1

    return tracks


# -----------------------------------------------------------------------------
# Consensus video + montage
# -----------------------------------------------------------------------------


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _voter_color_bgr(voters: int, n_trackers: int) -> tuple[int, int, int]:
    """Red -> Yellow -> Green based on voter fraction.

    2/4 = red, 3/4 = yellow, 4/4 = green.  Generalises to any n_trackers.
    """
    if n_trackers <= 1:
        return (0, 255, 0)
    frac = (voters - 1) / max(1, n_trackers - 1)
    frac = max(0.0, min(1.0, frac))
    if frac < 0.5:
        t = frac * 2.0
        return (0, int(255 * t), 255)           # red  -> yellow
    t = (frac - 0.5) * 2.0
    return (0, 255, int(255 * (1.0 - t)))        # yellow -> green


def render_consensus_video(
    input_video: Path,
    output_path: Path,
    pool_tracks: list[tuple[str, list[dict[str, Any]], int]],
    *,
    max_tail: int = 10,
) -> bool:
    """Render consensus tracks from multiple pools onto a single video.

    pool_tracks: list of (pool_name, linked_tracks, n_trackers_in_pool).
    Slow pool tracks get square markers, fast pool tracks get diamond markers.
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

    frame_idx = 0
    while True:
        ret, fr = cap.read()
        if not ret or fr is None:
            break
        canvas = _ensure_bgr(fr).copy()

        for pool_name, linked_tracks, n_pool in pool_tracks:
            is_fast_pool = pool_name.lower() == "fast"
            min_consensus_len = 3

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

                if frame_idx - last_frame_seen > max_tail:
                    continue

                if max_tail > 0 and len(tail) > max_tail:
                    tail = tail[-max_tail:]

                is_active = last_frame_seen == frame_idx

                line_thickness = 2 if is_fast_pool else 1
                if len(tail) > 1:
                    for i in range(len(tail) - 1):
                        _, _, v1 = tail[i + 1]
                        color = _voter_color_bgr(v1, n_pool)
                        cv2.line(canvas, tail[i][:2], tail[i + 1][:2], color, line_thickness, cv2.LINE_AA)

                if is_active:
                    hx, hy, hv = tail[-1]
                    color = _voter_color_bgr(hv, n_pool)
                    if is_fast_pool:
                        r = 4
                        pts = np.array([[hx, hy - r], [hx + r, hy], [hx, hy + r], [hx - r, hy]], dtype=np.int32)
                        cv2.polylines(canvas, [pts], True, color, 1, cv2.LINE_AA)
                    else:
                        cv2.circle(canvas, (hx, hy), 2, color, -1, cv2.LINE_AA)
                        cv2.rectangle(canvas, (hx - 3, hy - 3), (hx + 3, hy + 3), color, 1, cv2.LINE_AA)

        out.write(canvas)
        frame_idx += 1

    out.release()
    cap.release()
    return True


def _letterbox_into(frame: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    from tracking_core.comparison_video import _letterbox_into as lb

    return lb(frame, box_w, box_h)


def _extract_overlay_panel(frame: np.ndarray, reference_width: int) -> np.ndarray:
    from tracking_core.comparison_video import _extract_overlay_panel as ex

    return ex(frame, reference_width)


def write_ensemble_montage(
    original_video: Path,
    consensus_video: Path,
    member_videos: list[tuple[str, Path]],
    output_path: Path,
    *,
    top_label_h: int = 28,
    bottom_label_h: int = 22,
    gap: int = 8,
) -> bool:
    """Top: original | consensus. Bottom: four tracker overlay panels."""
    cap_o = cv2.VideoCapture(str(original_video))
    cap_c = cv2.VideoCapture(str(consensus_video))
    caps_m = [cv2.VideoCapture(str(p)) for _, p in member_videos]

    if not cap_o.isOpened() or not cap_c.isOpened() or any(not c.isOpened() for c in caps_m):
        cap_o.release()
        cap_c.release()
        for c in caps_m:
            c.release()
        return False

    fps = int(cap_o.get(cv2.CAP_PROP_FPS)) or int(cap_c.get(cv2.CAP_PROP_FPS)) or 25
    ret0, fr0 = cap_o.read()
    if not ret0 or fr0 is None:
        cap_o.release()
        cap_c.release()
        for c in caps_m:
            c.release()
        return False
    fr0 = _ensure_bgr(fr0)
    ref_w = fr0.shape[1]
    oh0, ow0 = fr0.shape[:2]

    # Top row: side-by-side original | consensus, each half canvas
    canvas_w = max(640, ref_w * 2)
    top_panel_w = canvas_w // 2
    top_h = max(1, int(round(top_panel_w * oh0 / max(1, ow0))))

    n = len(caps_m)
    cols = n
    cell_w = canvas_w // cols
    ret_m0, fr_m0 = caps_m[0].read()
    if not ret_m0 or fr_m0 is None:
        cap_o.release()
        cap_c.release()
        for c in caps_m:
            c.release()
        return False
    fr_m0 = _extract_overlay_panel(_ensure_bgr(fr_m0), ref_w)
    mh, mw = fr_m0.shape[:2]
    cell_aspect = mw / max(1.0, float(mh))
    cell_inner_h = max(72, int(round(cell_w / cell_aspect)))
    bottom_h = bottom_label_h + cell_inner_h + 4

    canvas_h = top_label_h + top_h + gap + bottom_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_w, canvas_h))
    if not writer.isOpened():
        cap_o.release()
        cap_c.release()
        for c in caps_m:
            c.release()
        return False

    cap_o.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_c.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for c in caps_m:
        c.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret_o, frame_o = cap_o.read()
        ret_c, frame_c = cap_c.read()
        if not ret_o or frame_o is None or not ret_c or frame_c is None:
            break
        frame_o = _ensure_bgr(frame_o)
        frame_c = _ensure_bgr(frame_c)

        member_frames: list[np.ndarray] = []
        ok = True
        for c in caps_m:
            ret, fr = c.read()
            if not ret or fr is None:
                ok = False
                break
            member_frames.append(_extract_overlay_panel(_ensure_bgr(fr), ref_w))
        if not ok:
            break

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        cv2.putText(
            canvas,
            "Original",
            (6, top_label_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Ensemble consensus",
            (top_panel_w + 6, top_label_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        left = cv2.resize(frame_o, (top_panel_w, top_h), interpolation=cv2.INTER_AREA)
        right = cv2.resize(frame_c, (top_panel_w, top_h), interpolation=cv2.INTER_AREA)
        canvas[top_label_h : top_label_h + top_h, 0:top_panel_w] = left
        canvas[top_label_h : top_label_h + top_h, top_panel_w : top_panel_w + top_panel_w] = right

        y0 = top_label_h + top_h + gap
        margin = 2
        for idx, (lab, _p) in enumerate(member_videos):
            x0 = idx * cell_w
            cv2.putText(
                canvas,
                lab[:28],
                (x0 + margin, y0 + bottom_label_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            inner_y = y0 + bottom_label_h
            inner_w = cell_w - 2 * margin
            inner_h = cell_inner_h
            cell_img = _letterbox_into(member_frames[idx], inner_w, inner_h)
            canvas[inner_y : inner_y + inner_h, x0 + margin : x0 + margin + inner_w] = cell_img

        writer.write(canvas)

    writer.release()
    cap_o.release()
    cap_c.release()
    for c in caps_m:
        c.release()
    return True


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    epilog = """
Config file:
  If ensemble_config.json exists next to this script, it is loaded automatically (unless
  --builtin-runs). Copy the template:  python run_ensemble.py --write-ensemble-config

Progress:
  Use --live-log for interleaved, prefixed tracker logs (frame progress every ~100 frames)
  and --status-interval to print which runs are still in flight.

Parameter sets:
  Edit the "runs" array: each entry has key, tracker_name, label, and overrides (dict of
  tracker kwargs merged onto the registry). See tracking_core/variants.py PMB_BASELINE_KWARGS.
"""
    p = argparse.ArgumentParser(
        description="Parallel PMB/PMBM ensemble run + montage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument(
        "--input-video",
        type=Path,
        default=None,
        help="Input video (required unless using --write-ensemble-config only).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run folder (default: <workspace>/_ensemble_output/<timestamp>)",
    )
    p.add_argument(
        "--ensemble-config",
        type=Path,
        default=None,
        help="JSON file with 'runs' (and optional 'fusion'). Overrides auto-loaded ensemble_config.json.",
    )
    p.add_argument(
        "--builtin-runs",
        action="store_true",
        help="Ignore ensemble_config.json; use built-in DEFAULT_ENSEMBLE_RUNS only.",
    )
    p.add_argument(
        "--write-ensemble-config",
        type=Path,
        nargs="?",
        const=Path("ensemble_config.json"),
        default=None,
        help="Write default ensemble_config.json template and exit (default path: ./ensemble_config.json).",
    )
    p.add_argument(
        "--quorum",
        type=int,
        default=None,
        help="Min distinct trackers agreeing (default 2, or fusion.quorum in config file)",
    )
    p.add_argument(
        "--spatial-threshold",
        type=float,
        default=None,
        help="Pixels for voting cluster radius (default 8, or fusion.spatial_threshold_px in config)",
    )
    p.add_argument(
        "--link-threshold",
        type=float,
        default=None,
        help="Max px to link consensus across frames (default: 2 * spatial-threshold)",
    )
    p.add_argument("--preset", type=str, default=None)
    p.add_argument("--preset-file", type=Path, default=None)
    p.add_argument("--verbose-trackers", action="store_true", help="Do not silence tracker stdout (no effect with --live-log)")
    p.add_argument(
        "--live-log",
        action="store_true",
        help="Stream each tracker's verbose progress with [run_key] prefixes (uses threads; slightly less CPU isolation than default process pool).",
    )
    p.add_argument(
        "--status-interval",
        type=float,
        default=15.0,
        help="With --live-log, print pending run keys every N seconds (0 disables). Default 15.",
    )
    p.add_argument("--workers", type=int, default=4)
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

    if args.write_ensemble_config is not None:
        cfg_out = Path(args.write_ensemble_config)
        if not cfg_out.is_absolute():
            cfg_out = workspace / cfg_out
        cfg_out.write_text(
            json.dumps(default_ensemble_config_document(), indent=2),
            encoding="utf-8",
        )
        print(f"Wrote default ensemble config template: {cfg_out}")
        return 0

    if args.input_video is None:
        print("--input-video is required (unless using --write-ensemble-config).", file=sys.stderr)
        return 2

    input_video = args.input_video.resolve()
    if not input_video.is_file():
        print(f"Input video not found: {input_video}", file=sys.stderr)
        return 1

    try:
        runs, fusion, config_path = resolve_ensemble_runs(
            workspace,
            ensemble_config=args.ensemble_config,
            builtin_runs=args.builtin_runs,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Ensemble config error: {exc}", file=sys.stderr)
        return 1

    quorum = args.quorum if args.quorum is not None else int(fusion.get("quorum", 2))
    spatial_threshold = (
        args.spatial_threshold
        if args.spatial_threshold is not None
        else float(fusion.get("spatial_threshold_px", 8.0))
    )
    link_thr = args.link_threshold
    if link_thr is None and fusion.get("link_threshold_px") is not None:
        link_thr = float(fusion["link_threshold_px"])
    if link_thr is None:
        link_thr = spatial_threshold * 2.0

    run_dir = args.output_dir
    if run_dir is None:
        run_dir = workspace / "_ensemble_output" / time.strftime("ensemble_%Y%m%d_%H%M%S")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    preset_file_str = str(args.preset_file.resolve()) if args.preset_file else None

    from tracking_core.variants import TRACKER_VARIANTS

    registry_names = {s["name"] for s in TRACKER_VARIANTS}
    for r in runs:
        if r["tracker_name"] not in registry_names:
            print(
                f"Warning: tracker_name {r['tracker_name']!r} is not in TRACKER_VARIANTS.",
                file=sys.stderr,
            )

    jobs: list[dict[str, Any]] = []
    for cfg in runs:
        sub = run_dir / cfg["key"]
        jobs.append(
            {
                "workspace_root": str(workspace),
                "ensemble_key": cfg["key"],
                "tracker_name": cfg["tracker_name"],
                "input_video": str(input_video),
                "output_dir": str(sub),
                "overrides": cfg["overrides"],
                "preset_name": args.preset,
                "preset_file": preset_file_str,
                "quiet": not args.verbose_trackers and not args.live_log,
            }
        )

    resolved_snapshot = {
        "config_path": str(config_path) if config_path else None,
        "builtin_only": args.builtin_runs,
        "runs": deepcopy(runs),
        "fusion_effective": {
            "quorum": quorum,
            "spatial_threshold_px": spatial_threshold,
            "link_threshold_px": link_thr,
        },
    }
    (run_dir / "ensemble_config_resolved.json").write_text(
        json.dumps(resolved_snapshot, indent=2),
        encoding="utf-8",
    )

    print(f"Output directory: {run_dir}", flush=True)
    if config_path:
        print(f"Ensemble config:  {config_path}", flush=True)
    elif args.builtin_runs:
        print("Ensemble config:  built-in DEFAULT_ENSEMBLE_RUNS", flush=True)
    else:
        print("Ensemble config:  built-in defaults (no ensemble_config.json found)", flush=True)
    print(
        f"Fusion: quorum={quorum}, spatial_threshold_px={spatial_threshold}, link_threshold_px={link_thr}",
        flush=True,
    )
    mode = "live-log (threads + prefixed verbose)" if args.live_log else f"process pool ({args.workers} workers)"
    print(f"Running {len(jobs)} trackers in parallel — {mode}", flush=True)

    results: list[dict[str, Any]] = []
    future_to_key: dict[Any, str] = {}
    stop_status = threading.Event()

    def status_loop() -> None:
        while True:
            if stop_status.wait(timeout=args.status_interval):
                break
            pending = [future_to_key[f] for f in list(future_to_key.keys()) if not f.done()]
            if pending:
                print(
                    f"[ensemble] Still running ({len(pending)}): {', '.join(sorted(pending))}",
                    flush=True,
                )

    status_thread: threading.Thread | None = None
    if args.status_interval > 0:
        status_thread = threading.Thread(target=status_loop, daemon=True)

    try:
        if args.live_log:
            n_workers = min(args.workers, len(jobs))
            with ThreadPoolExecutor(max_workers=max(1, n_workers)) as ex:
                for j in jobs:
                    fut = ex.submit(_run_live_monitored_job, j)
                    future_to_key[fut] = j["ensemble_key"]
                if status_thread is not None:
                    status_thread.start()
                for fut in as_completed(future_to_key):
                    job_key = future_to_key[fut]
                    try:
                        summ = fut.result()
                        results.append(summ)
                        print(f"[ensemble] Finished {job_key} -> {summ.get('status', '?')}", flush=True)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[ensemble] {job_key} -> exception: {exc}", file=sys.stderr, flush=True)
                        results.append({"ensemble_key": job_key, "status": "error", "error": str(exc)})
        else:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
                for j in jobs:
                    fut = ex.submit(_ensemble_tracker_job, j)
                    future_to_key[fut] = j["ensemble_key"]
                if status_thread is not None:
                    status_thread.start()
                for fut in as_completed(future_to_key):
                    job_key = future_to_key[fut]
                    try:
                        summ = fut.result()
                        results.append(summ)
                        st = summ.get("status", "?")
                        print(f"  [{job_key}] -> {st}", flush=True)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [{job_key}] -> exception: {exc}", file=sys.stderr, flush=True)
                        results.append({"ensemble_key": job_key, "status": "error", "error": str(exc)})
    finally:
        stop_status.set()
        if status_thread is not None and status_thread.is_alive():
            status_thread.join(timeout=2.0)

    failed = [r for r in results if r.get("status") != "ok"]
    if failed:
        print("Some tracker runs failed; consensus may be incomplete.", file=sys.stderr, flush=True)

    pools: dict[str, list[int]] = defaultdict(list)
    member_videos: list[tuple[str, Path]] = []
    all_entries: list[tuple[int, list[tuple[int, int, float, float]]]] = []
    run_index_to_pool: dict[int, str] = {}

    for idx, cfg in enumerate(runs):
        sub = run_dir / cfg["key"]
        tlog = sub / "tracks.txt"
        pool_name = cfg.get("pool", "default")
        if tlog.is_file():
            tidx = len(all_entries)
            all_entries.append((tidx, parse_pmb_track_entries(tlog)))
            pools[pool_name].append(tidx)
            run_index_to_pool[tidx] = pool_name
        vid = sub / "output_video.mp4"
        if vid.is_file():
            member_videos.append((cfg["label"], vid))

    pool_tracks: list[tuple[str, list[dict[str, Any]], int]] = []
    pool_stats: dict[str, dict[str, int]] = {}

    pool_fusion_overrides = fusion.get("pool_overrides") or {}

    pool_names_sorted = sorted(pools.keys())
    for pool_name in pool_names_sorted:
        pool_indices = pools[pool_name]
        pool_entries = [(seq, entries) for seq, entries in all_entries if seq in pool_indices]
        renumbered = [(i, entries) for i, (_seq, entries) in enumerate(pool_entries)]
        n_pool = len(pool_entries)

        pf = pool_fusion_overrides.get(pool_name, {})
        pool_quorum = int(pf.get("quorum", quorum))
        pool_spatial = float(pf.get("spatial_threshold_px", spatial_threshold))
        pool_link = float(pf.get("link_threshold_px", link_thr))

        print(
            f"[ensemble] Pool '{pool_name}': {n_pool} trackers, quorum={pool_quorum}, "
            f"spatial={pool_spatial}px, link={pool_link}px",
            flush=True,
        )

        consensus_by_frame = build_per_frame_consensus(
            renumbered,
            spatial_threshold=pool_spatial,
            quorum=pool_quorum,
            n_trackers=n_pool,
        )
        linked = link_consensus_across_frames(consensus_by_frame, link_threshold=pool_link)
        pool_tracks.append((pool_name, linked, n_pool))
        pool_stats[pool_name] = {
            "n_trackers": n_pool,
            "quorum": pool_quorum,
            "spatial_threshold_px": pool_spatial,
            "link_threshold_px": pool_link,
            "consensus_frames": len(consensus_by_frame),
            "consensus_tracks": len(linked),
        }
        print(f"           {len(consensus_by_frame)} frames with votes, {len(linked)} consensus tracks", flush=True)

    print(f"[ensemble] Building consensus overlay...", flush=True)
    consensus_path = run_dir / "consensus_overlay.mp4"
    if not render_consensus_video(input_video, consensus_path, pool_tracks):
        print("Failed to write consensus_overlay.mp4", file=sys.stderr, flush=True)
        return 1

    montage_path = run_dir / "ensemble_montage.mp4"
    if len(member_videos) >= 1 and consensus_path.is_file():
        print(f"[ensemble] Building montage ({len(member_videos)} member panel(s))...", flush=True)
        ok_m = write_ensemble_montage(
            input_video,
            consensus_path,
            member_videos,
            montage_path,
        )
        if not ok_m:
            print("Failed to write ensemble_montage.mp4", file=sys.stderr, flush=True)
    else:
        print(
            "Skipping montage (need at least one member video and consensus).",
            file=sys.stderr,
            flush=True,
        )

    total_consensus_tracks = sum(s["consensus_tracks"] for s in pool_stats.values())
    manifest = {
        "input_video": str(input_video),
        "output_dir": str(run_dir),
        "ensemble_config_path": str(config_path) if config_path else None,
        "quorum": quorum,
        "spatial_threshold_px": spatial_threshold,
        "link_threshold_px": link_thr,
        "live_log": args.live_log,
        "tracker_results": results,
        "pools": pool_stats,
        "consensus_tracks_total": total_consensus_tracks,
        "consensus_overlay": str(consensus_path),
        "ensemble_montage": str(montage_path) if montage_path.is_file() else None,
    }
    (run_dir / "ensemble_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDone.", flush=True)
    print(f"  Consensus video: {consensus_path}", flush=True)
    if montage_path.is_file():
        print(f"  Montage:         {montage_path}", flush=True)
    print(f"  Manifest:        {run_dir / 'ensemble_manifest.json'}", flush=True)
    print(f"  Resolved config: {run_dir / 'ensemble_config_resolved.json'}", flush=True)
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
