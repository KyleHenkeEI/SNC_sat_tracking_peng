#!/usr/bin/env python3
"""
Interactive consensus model selector with analytics dashboard.

Browse individual tracker results in a grid, compare side-by-side with the
raw video, explore parameter-space distributions and statistics, select the
runs you want, and build a consensus video from the chosen models.

Usage:
    python select_consensus.py --run-dir _txtpmb_fullgrid/run_20260401_123456
    python select_consensus.py --run-dir _txtpmb_ensemble/run_20260401_051643 --input-video raw.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("tkinter is required. It ships with standard CPython on Windows.", file=sys.stderr)
    sys.exit(1)

from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.cm as mpl_cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

try:
    from scipy.stats import gaussian_kde as _gaussian_kde
except ImportError:
    _gaussian_kde = None  # type: ignore[assignment,misc]


METRIC_CHOICES = [
    "tracks_metric", "total_detections", "runtime_s",
    "mean_confidence", "mean_speed", "mean_lifetime",
]
PARAM_CHOICES = [
    "bg_threshold", "min_intensity", "clutter_rate",
    "existence_threshold", "detection_prob",
    "max_speed", "max_acceleration", "max_direction_change",
]
ALL_SORT_OPTIONS = METRIC_CHOICES + PARAM_CHOICES
TRACK_DEPENDENT = {"mean_confidence", "mean_speed", "mean_lifetime"}


# ---------------------------------------------------------------------------
# Track-level statistics (lazy parser for tracks.txt)
# ---------------------------------------------------------------------------

class TrackStats:
    """Lazy parser for tracks.txt that computes per-run distributions."""

    def __init__(self, track_log: Path):
        self._path = track_log
        self._parsed = False
        self.n_entries = 0
        self.n_unique_tracks = 0

    def ensure_parsed(self) -> bool:
        if self._parsed:
            return self.n_entries > 0
        self._parsed = True
        if not self._path.is_file():
            self._set_empty()
            return False
        try:
            self._parse()
            return self.n_entries > 0
        except Exception:
            self._set_empty()
            return False

    def _set_empty(self) -> None:
        for attr in ("frames", "track_ids", "xs", "ys", "existence_probs",
                      "confidences", "speeds", "ages", "detections_arr",
                      "track_lifetimes"):
            setattr(self, attr, np.array([]))
        self.n_entries = 0
        self.n_unique_tracks = 0
        for attr in ("confidence_mean", "confidence_std", "confidence_median",
                      "speed_mean", "speed_std", "speed_median",
                      "existence_mean", "existence_std",
                      "lifetime_mean", "lifetime_std", "lifetime_median"):
            setattr(self, attr, 0.0)
        self.confidence_iqr = (0.0, 0.0)
        self.speed_iqr = (0.0, 0.0)
        self.spatial_bbox = (0.0, 0.0, 0.0, 0.0)

    def _parse(self) -> None:
        rows: list[tuple] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line[0] in ("#", "="):
                    continue
                parts = line.split(",")
                if len(parts) < 9:
                    continue
                try:
                    rows.append((
                        int(parts[0]), int(parts[1]),
                        float(parts[2]), float(parts[3]),
                        float(parts[4]), float(parts[5]),
                        float(parts[6]), int(parts[7]), int(parts[8]),
                    ))
                except (ValueError, IndexError):
                    continue

        if not rows:
            self._set_empty()
            return

        data = np.array(rows)
        self.frames = data[:, 0].astype(int)
        self.track_ids = data[:, 1].astype(int)
        self.xs = data[:, 2]
        self.ys = data[:, 3]
        self.existence_probs = data[:, 4]
        self.confidences = data[:, 5]
        self.speeds = data[:, 6]
        self.ages = data[:, 7].astype(int)
        self.detections_arr = data[:, 8].astype(int)
        self.n_entries = len(rows)

        unique_tids = np.unique(self.track_ids)
        self.n_unique_tracks = len(unique_tids)
        lifetimes = []
        for tid in unique_tids:
            f = self.frames[self.track_ids == tid]
            lifetimes.append(int(f.max() - f.min() + 1))
        self.track_lifetimes = np.array(lifetimes, dtype=int)

        self._compute_stats(self.confidences, "confidence")
        self._compute_stats(self.speeds, "speed")
        self.existence_mean = float(np.mean(self.existence_probs))
        self.existence_std = float(np.std(self.existence_probs))

        if len(self.track_lifetimes) > 0:
            self.lifetime_mean = float(np.mean(self.track_lifetimes))
            self.lifetime_std = float(np.std(self.track_lifetimes))
            self.lifetime_median = float(np.median(self.track_lifetimes))
        else:
            self.lifetime_mean = self.lifetime_std = self.lifetime_median = 0.0

        self.spatial_bbox = (
            float(self.xs.min()), float(self.ys.min()),
            float(self.xs.max()), float(self.ys.max()),
        )

    def _compute_stats(self, arr: np.ndarray, prefix: str) -> None:
        setattr(self, f"{prefix}_mean", float(np.mean(arr)))
        setattr(self, f"{prefix}_std", float(np.std(arr)))
        setattr(self, f"{prefix}_median", float(np.median(arr)))
        q25, q75 = np.percentile(arr, [25, 75])
        setattr(self, f"{prefix}_iqr", (float(q25), float(q75)))


# ---------------------------------------------------------------------------
# Run-folder discovery
# ---------------------------------------------------------------------------

class RunInfo:
    """Metadata for a single tracker run subfolder."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.key = folder.name
        self.video_path = folder / "output_video.mp4"
        self.track_log = folder / "tracks.txt"
        self.summary_path = folder / "summary.json"
        self.overrides: dict[str, Any] = {}
        self.label = self.key
        self.selected = False
        self.tracks_metric: int = 0
        self.total_detections: int = 0
        self.frames_processed: int = 0
        self.runtime_s: float = 0.0
        self.tracker_name: str = ""
        self.status: str = ""
        self._track_stats: TrackStats | None = None
        self._load_meta()

    def _load_meta(self) -> None:
        if self.summary_path.is_file():
            try:
                data = json.loads(self.summary_path.read_text(encoding="utf-8"))
                self.overrides = data.get("overrides_applied", {})
                self.tracks_metric = int(data.get("tracks_metric", 0) or 0)
                self.total_detections = int(data.get("total_detections", 0) or 0)
                self.frames_processed = int(data.get("frames_processed", 0) or 0)
                self.runtime_s = float(data.get("runtime_s", 0.0) or 0.0)
                self.tracker_name = data.get("tracker_name", "")
                self.status = data.get("status", "")
            except (json.JSONDecodeError, OSError):
                pass

        bg = self.overrides.get("bg_threshold", "?")
        mi = self.overrides.get("min_intensity", "?")
        cr = self.overrides.get("clutter_rate", "?")
        if bg != "?" or mi != "?" or cr != "?":
            self.label = f"bg={bg}  mi={mi}  cr={cr}"
        else:
            self.label = self.key

    @property
    def has_video(self) -> bool:
        return self.video_path.is_file()

    @property
    def has_tracks(self) -> bool:
        return self.track_log.is_file()

    @property
    def track_stats(self) -> TrackStats:
        if self._track_stats is None:
            self._track_stats = TrackStats(self.track_log)
        return self._track_stats

    def get_metric(self, name: str) -> float:
        """Return a named metric value for sorting/coloring/plotting."""
        if name == "tracks_metric":
            return float(self.tracks_metric)
        if name == "total_detections":
            return float(self.total_detections)
        if name == "runtime_s":
            return self.runtime_s
        if name == "mean_confidence":
            ts = self.track_stats
            return ts.confidence_mean if ts._parsed and ts.n_entries > 0 else 0.0
        if name == "mean_speed":
            ts = self.track_stats
            return ts.speed_mean if ts._parsed and ts.n_entries > 0 else 0.0
        if name == "mean_lifetime":
            ts = self.track_stats
            return ts.lifetime_mean if ts._parsed and ts.n_entries > 0 else 0.0
        val = self.overrides.get(name)
        return float(val) if val is not None else 0.0


def discover_runs(run_dir: Path) -> list[RunInfo]:
    """Find all sub-folders that contain tracker outputs."""
    runs: list[RunInfo] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        info = RunInfo(child)
        if info.has_video or info.has_tracks:
            runs.append(info)
    return runs


def resolve_input_video(run_dir: Path, cli_path: Path | None) -> Path | None:
    if cli_path and cli_path.is_file():
        return cli_path.resolve()
    for name in ("txtpmb_manifest.json", "ensemble_manifest.json"):
        mf = run_dir / name
        if mf.is_file():
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
                p = Path(data.get("input_video", ""))
                if p.is_file():
                    return p.resolve()
            except (json.JSONDecodeError, OSError):
                pass
    return None


# ---------------------------------------------------------------------------
# Ensemble analytics (aggregation + bootstrap CI)
# ---------------------------------------------------------------------------

class EnsembleAnalytics:
    """Aggregate statistics across all runs for distribution / explorer views."""

    def __init__(self, runs: list[RunInfo]):
        self.runs = runs
        self._tracks_loaded = False

    def load_all_tracks(self, progress_cb=None) -> None:
        """Parse every run's tracks.txt (call from background thread)."""
        if self._tracks_loaded:
            return
        for i, run in enumerate(self.runs):
            run.track_stats.ensure_parsed()
            if progress_cb:
                progress_cb(i + 1, len(self.runs))
        self._tracks_loaded = True

    def get_metric_array(self, metric: str,
                         runs: list[RunInfo] | None = None) -> np.ndarray:
        target = runs if runs is not None else self.runs
        return np.array([r.get_metric(metric) for r in target])

    def get_param_values(self, param: str) -> list[float]:
        vals: set[float] = set()
        for r in self.runs:
            v = r.overrides.get(param)
            if v is not None:
                vals.add(float(v))
        return sorted(vals)

    def group_by_param(self, param: str, metric: str,
                       runs: list[RunInfo] | None = None
                       ) -> dict[float, np.ndarray]:
        target = runs if runs is not None else self.runs
        groups: dict[float, list[float]] = {}
        for r in target:
            pv = r.overrides.get(param)
            if pv is None:
                continue
            groups.setdefault(float(pv), []).append(r.get_metric(metric))
        return {k: np.array(v) for k, v in sorted(groups.items())}

    @staticmethod
    def bootstrap_ci(data: np.ndarray, n_boot: int = 2000,
                     ci: float = 0.95) -> tuple[float, float, float]:
        """Returns (mean, ci_low, ci_high)."""
        if len(data) == 0:
            return 0.0, 0.0, 0.0
        if len(data) < 2:
            m = float(data[0])
            return m, m, m
        rng = np.random.default_rng(42)
        means = np.array([
            float(np.mean(rng.choice(data, size=len(data), replace=True)))
            for _ in range(n_boot)
        ])
        alpha = (1 - ci) / 2
        lo = float(np.percentile(means, 100 * alpha))
        hi = float(np.percentile(means, 100 * (1 - alpha)))
        return float(np.mean(data)), lo, hi

    def param_level_ci(self, param: str, metric: str,
                       runs: list[RunInfo] | None = None
                       ) -> list[tuple[float, float, float, float]]:
        """(level, mean, ci_lo, ci_hi) per parameter level."""
        groups = self.group_by_param(param, metric, runs)
        results: list[tuple[float, float, float, float]] = []
        for level, vals in groups.items():
            m, lo, hi = self.bootstrap_ci(vals)
            results.append((level, m, lo, hi))
        return results

    def available_params(self) -> list[str]:
        """Return parameter keys that actually vary across runs."""
        result: list[str] = []
        for p in PARAM_CHOICES:
            vals: set = set()
            for r in self.runs:
                v = r.overrides.get(p)
                if v is not None:
                    vals.add(v)
            if len(vals) > 1:
                result.append(p)
        return result if result else PARAM_CHOICES[:3]


# ---------------------------------------------------------------------------
# Thumbnail helpers
# ---------------------------------------------------------------------------

THUMB_SIZE = 160


def extract_thumbnail(video_path: Path,
                      target_frame: int | None = None) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seek = target_frame if target_frame is not None else max(0, total // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(seek, max(0, total - 1)))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]
    scale = THUMB_SIZE / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((THUMB_SIZE, THUMB_SIZE, 3), dtype=np.uint8)
    y0 = (THUMB_SIZE - new_h) // 2
    x0 = (THUMB_SIZE - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = frame
    return canvas


def cv2_to_tk(bgr: np.ndarray) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


def add_border(bgr: np.ndarray, color_bgr: tuple[int, int, int],
               thickness: int = 4) -> np.ndarray:
    bordered = bgr.copy()
    cv2.rectangle(bordered, (0, 0),
                  (bordered.shape[1] - 1, bordered.shape[0] - 1),
                  color_bgr, thickness)
    return bordered


def metric_color_bgr(value: float, vmin: float, vmax: float
                     ) -> tuple[int, int, int]:
    """Map a scalar value to a BGR color on the RdYlGn colormap."""
    if vmax <= vmin:
        norm_val = 0.5
    else:
        norm_val = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    rgba = mpl_cm.RdYlGn(norm_val)
    return (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))


# ---------------------------------------------------------------------------
# Tooltip helper
# ---------------------------------------------------------------------------

class ToolTip:
    """Hover tooltip for tkinter widgets."""

    def __init__(self, widget: tk.Widget, text_func):
        self.widget = widget
        self.text_func = text_func
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<Button-1>", self._hide, add="+")

    def _show(self, event=None) -> None:
        text = self.text_func()
        if not text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(tw, text=text, justify=tk.LEFT,
                       background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                       font=("Consolas", 8), padx=4, pady=2)
        lbl.pack()

    def _hide(self, event=None) -> None:
        if self._tip:
            self._tip.destroy()
            self._tip = None


# ---------------------------------------------------------------------------
# Comparison playback window (with stats sidebar)
# ---------------------------------------------------------------------------

class CompareWindow:
    """Side-by-side playback: raw video (left) vs overlay (right) + stats."""

    DISPLAY_W = 320
    STATS_W = 380

    def __init__(self, parent: tk.Tk, run_info: RunInfo,
                 input_video: Path, on_toggle: Any):
        self.run = run_info
        self.input_video = input_video
        self.on_toggle = on_toggle
        self.playing = False
        self._after_id: str | None = None
        self._updating_slider = False

        self.cap_raw = cv2.VideoCapture(str(input_video))
        self.cap_ovl = cv2.VideoCapture(str(run_info.video_path))
        self.fps = int(self.cap_raw.get(cv2.CAP_PROP_FPS)) or 15
        self.total_frames = int(self.cap_raw.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.speed = 1.0

        self.win = tk.Toplevel(parent)
        self.win.title(f"Compare: {run_info.label}")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- Top: video panels + stats sidebar ---
        top_frame = tk.Frame(self.win)
        top_frame.pack(fill=tk.BOTH, expand=True)

        video_frame = tk.Frame(top_frame)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.label_raw = tk.Label(video_frame)
        self.label_raw.pack(side=tk.LEFT, padx=2, pady=2)
        self.label_ovl = tk.Label(video_frame)
        self.label_ovl.pack(side=tk.LEFT, padx=2, pady=2)

        self._build_stats_sidebar(top_frame, run_info)

        # --- Controls ---
        ctrl_frame = tk.Frame(self.win)
        ctrl_frame.pack(fill=tk.X, padx=6, pady=4)

        self.btn_play = ttk.Button(ctrl_frame, text="Play",
                                   command=self._toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="<< Step",
                   command=lambda: self._step(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="Step >>",
                   command=lambda: self._step(1)).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl_frame, text="Speed:").pack(side=tk.LEFT, padx=(10, 2))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_box = ttk.Combobox(ctrl_frame, textvariable=self.speed_var,
                                 values=[0.25, 0.5, 1.0, 2.0, 4.0], width=5)
        speed_box.pack(side=tk.LEFT, padx=2)
        speed_box.bind("<<ComboboxSelected>>",
                       lambda _: self._update_speed())

        self.sel_var = tk.BooleanVar(value=run_info.selected)
        self.btn_sel = ttk.Checkbutton(
            ctrl_frame, text="Include in consensus", variable=self.sel_var,
            command=self._toggle_selection,
        )
        self.btn_sel.pack(side=tk.RIGHT, padx=6)

        # --- Slider ---
        slider_frame = tk.Frame(self.win)
        slider_frame.pack(fill=tk.X, padx=6, pady=(0, 4))
        self.slider = ttk.Scale(
            slider_frame, from_=0, to=max(0, self.total_frames - 1),
            orient=tk.HORIZONTAL, command=self._on_slider,
        )
        self.slider.pack(fill=tk.X)
        self.frame_label = ttk.Label(slider_frame, text="Frame 0 / 0")
        self.frame_label.pack()

        self._show_frame(0)

    # ---- Stats sidebar ----

    def _build_stats_sidebar(self, parent: tk.Frame,
                             run: RunInfo) -> None:
        sf = tk.Frame(parent, width=self.STATS_W)
        sf.pack(side=tk.RIGHT, fill=tk.Y, padx=4, pady=4)
        sf.pack_propagate(False)

        tk.Label(sf, text="Run Statistics",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=4,
                                                      pady=(4, 0))

        lines = [
            f"Tracker: {run.tracker_name}",
            f"Status:  {run.status}",
            f"Tracks:  {run.tracks_metric}",
            f"Detections: {run.total_detections}",
            f"Frames:  {run.frames_processed}",
            f"Runtime: {run.runtime_s:.1f}s",
            "--- Parameters ---",
        ]
        for k, v in sorted(run.overrides.items()):
            lines.append(f"  {k}: {v}")

        txt = tk.Text(sf, height=12, width=38, font=("Consolas", 8),
                      wrap=tk.WORD, relief=tk.GROOVE, bd=1)
        txt.pack(padx=4, pady=4, fill=tk.X)
        txt.insert(tk.END, "\n".join(lines))
        txt.configure(state=tk.DISABLED)

        self._stats_fig = Figure(figsize=(3.8, 4.5), dpi=100)
        self._stats_canvas = FigureCanvasTkAgg(self._stats_fig, master=sf)
        self._stats_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                                 padx=2, pady=2)

        def _bg_load():
            run.track_stats.ensure_parsed()
            try:
                self.win.after(0, self._draw_stats_charts)
            except Exception:
                pass

        threading.Thread(target=_bg_load, daemon=True).start()

    def _draw_stats_charts(self) -> None:
        ts = self.run.track_stats
        fig = self._stats_fig
        fig.clf()

        if ts.n_entries == 0:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "No track data available",
                    ha="center", va="center", fontsize=11, color="gray",
                    transform=ax.transAxes)
            ax.set_axis_off()
            self._stats_canvas.draw()
            return

        # 1 — Confidence histogram with 95 % CI band
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.hist(ts.confidences, bins=30, color="steelblue", alpha=0.7,
                 density=True, edgecolor="white", linewidth=0.3)
        ax1.axvline(ts.confidence_mean, color="red", ls="--", lw=1,
                    label=f"\u03bc={ts.confidence_mean:.3f}")
        n = len(ts.confidences)
        se = ts.confidence_std / np.sqrt(n) if n > 1 else 0
        ax1.axvspan(ts.confidence_mean - 1.96 * se,
                    ts.confidence_mean + 1.96 * se,
                    alpha=0.15, color="red", label="95% CI")
        ax1.set_title("Confidence", fontsize=8)
        ax1.legend(fontsize=5, loc="upper left")
        ax1.tick_params(labelsize=6)

        # 2 — Speed histogram
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.hist(ts.speeds, bins=30, color="coral", alpha=0.7,
                 density=True, edgecolor="white", linewidth=0.3)
        ax2.axvline(ts.speed_mean, color="red", ls="--", lw=1,
                    label=f"\u03bc={ts.speed_mean:.2f}")
        se_s = ts.speed_std / np.sqrt(n) if n > 1 else 0
        ax2.axvspan(ts.speed_mean - 1.96 * se_s,
                    ts.speed_mean + 1.96 * se_s,
                    alpha=0.15, color="red", label="95% CI")
        ax2.set_title("Speed", fontsize=8)
        ax2.legend(fontsize=5, loc="upper right")
        ax2.tick_params(labelsize=6)

        # 3 — Track lifetime
        ax3 = fig.add_subplot(2, 2, 3)
        if len(ts.track_lifetimes) > 0:
            bins = min(30, max(1, len(ts.track_lifetimes) // 3))
            ax3.hist(ts.track_lifetimes, bins=max(bins, 1),
                     color="seagreen", alpha=0.7, edgecolor="white",
                     linewidth=0.3)
            ax3.axvline(ts.lifetime_mean, color="red", ls="--", lw=1,
                        label=f"\u03bc={ts.lifetime_mean:.1f}")
            ax3.legend(fontsize=5)
        ax3.set_title("Track Lifetime (frames)", fontsize=8)
        ax3.tick_params(labelsize=6)

        # 4 — Spatial scatter
        ax4 = fig.add_subplot(2, 2, 4)
        cap = min(5000, len(ts.xs))
        if cap < len(ts.xs):
            idx = np.random.default_rng(0).choice(len(ts.xs), cap,
                                                   replace=False)
            ax4.scatter(ts.xs[idx], ts.ys[idx], s=1, alpha=0.3,
                        c="steelblue")
        else:
            ax4.scatter(ts.xs, ts.ys, s=1, alpha=0.3, c="steelblue")
        ax4.set_title("Spatial Coverage", fontsize=8)
        ax4.tick_params(labelsize=6)
        ax4.set_aspect("equal", adjustable="datalim")
        ax4.invert_yaxis()

        fig.tight_layout()
        self._stats_canvas.draw()

    # ---- Video playback (original logic preserved) ----

    def _read_pair(self, idx: int
                   ) -> tuple[np.ndarray | None, np.ndarray | None]:
        self.cap_raw.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret_r, fr_r = self.cap_raw.read()
        self.cap_ovl.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret_o, fr_o = self.cap_ovl.read()
        raw = fr_r if ret_r and fr_r is not None else None
        ovl = None
        if ret_o and fr_o is not None:
            w = fr_o.shape[1]
            ovl = fr_o[:, w // 2:]
        return raw, ovl

    def _resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        h, w = frame.shape[:2]
        scale = self.DISPLAY_W / max(w, 1)
        new_h = int(h * scale)
        return cv2.resize(frame, (self.DISPLAY_W, new_h),
                          interpolation=cv2.INTER_AREA)

    def _show_frame(self, idx: int) -> None:
        idx = max(0, min(idx, self.total_frames - 1))
        self.current_frame = idx
        raw, ovl = self._read_pair(idx)

        if raw is not None:
            disp_r = self._resize_for_display(raw)
            self._img_raw = cv2_to_tk(disp_r)
            self.label_raw.configure(image=self._img_raw)
        if ovl is not None:
            disp_o = self._resize_for_display(ovl)
            self._img_ovl = cv2_to_tk(disp_o)
            self.label_ovl.configure(image=self._img_ovl)

        self._updating_slider = True
        self.slider.set(idx)
        self._updating_slider = False
        self.frame_label.configure(
            text=f"Frame {idx} / {self.total_frames - 1}")

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.btn_play.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._play_loop()

    def _play_loop(self) -> None:
        if not self.playing:
            return
        nxt = self.current_frame + 1
        if nxt >= self.total_frames:
            self.playing = False
            self.btn_play.configure(text="Play")
            return
        self._show_frame(nxt)
        delay = max(1, int(1000.0 / (self.fps * self.speed)))
        self._after_id = self.win.after(delay, self._play_loop)

    def _step(self, delta: int) -> None:
        self.playing = False
        self.btn_play.configure(text="Play")
        self._show_frame(self.current_frame + delta)

    def _on_slider(self, val: str) -> None:
        if self._updating_slider:
            return
        self._show_frame(int(float(val)))

    def _update_speed(self) -> None:
        self.speed = max(0.1, self.speed_var.get())

    def _toggle_selection(self) -> None:
        self.run.selected = self.sel_var.get()
        if self.on_toggle:
            self.on_toggle()

    def _on_close(self) -> None:
        self.playing = False
        if self._after_id:
            try:
                self.win.after_cancel(self._after_id)
            except Exception:
                pass
        self.cap_raw.release()
        self.cap_ovl.release()
        self.win.destroy()


# ---------------------------------------------------------------------------
# Results playback window (raw + consensus top, members bottom)
# ---------------------------------------------------------------------------

class ResultsWindow:
    """Synchronized playback: raw | consensus on top, member videos below."""

    TOP_W = 320
    MEMBER_W = 200

    def __init__(
        self,
        parent: tk.Tk,
        input_video: Path,
        consensus_video: Path,
        members: list[tuple[str, Path]],
    ):
        self.playing = False
        self._after_id: str | None = None
        self._updating_slider = False

        self.cap_raw = cv2.VideoCapture(str(input_video))
        self.cap_con = cv2.VideoCapture(str(consensus_video))
        self.caps_mem: list[cv2.VideoCapture] = []
        self.member_labels_text: list[str] = []
        for label, vpath in members:
            self.caps_mem.append(cv2.VideoCapture(str(vpath)))
            self.member_labels_text.append(label)

        self.fps = int(self.cap_raw.get(cv2.CAP_PROP_FPS)) or 15
        self.total_frames = int(self.cap_raw.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.speed = 1.0

        self.win = tk.Toplevel(parent)
        self.win.title("Consensus Results")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        top_frame = tk.Frame(self.win)
        top_frame.pack(fill=tk.X, padx=4, pady=4)

        raw_col = tk.Frame(top_frame)
        raw_col.pack(side=tk.LEFT, padx=4)
        tk.Label(raw_col, text="Raw Video",
                 font=("Segoe UI", 9, "bold")).pack()
        self.lbl_raw = tk.Label(raw_col)
        self.lbl_raw.pack()

        con_col = tk.Frame(top_frame)
        con_col.pack(side=tk.LEFT, padx=4)
        tk.Label(con_col, text="Consensus",
                 font=("Segoe UI", 9, "bold")).pack()
        self.lbl_con = tk.Label(con_col)
        self.lbl_con.pack()

        ctrl = tk.Frame(self.win)
        ctrl.pack(fill=tk.X, padx=6, pady=4)

        self.btn_play = ttk.Button(ctrl, text="Play",
                                   command=self._toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="<< Step",
                   command=lambda: self._step(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="Step >>",
                   command=lambda: self._step(1)).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="Speed:").pack(side=tk.LEFT, padx=(10, 2))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_box = ttk.Combobox(ctrl, textvariable=self.speed_var,
                                 values=[0.25, 0.5, 1.0, 2.0, 4.0], width=5)
        speed_box.pack(side=tk.LEFT, padx=2)
        speed_box.bind("<<ComboboxSelected>>",
                       lambda _: self._update_speed())

        slider_frame = tk.Frame(self.win)
        slider_frame.pack(fill=tk.X, padx=6)
        self.slider = ttk.Scale(
            slider_frame, from_=0, to=max(0, self.total_frames - 1),
            orient=tk.HORIZONTAL, command=self._on_slider,
        )
        self.slider.pack(fill=tk.X)
        self.frame_label = ttk.Label(slider_frame, text="Frame 0 / 0")
        self.frame_label.pack()

        sep = ttk.Separator(self.win, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, padx=4, pady=4)

        tk.Label(self.win, text="Selected Parameter Sets",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8)

        bot_container = tk.Frame(self.win)
        bot_container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        bot_canvas = tk.Canvas(bot_container, height=self.MEMBER_W + 40)
        bot_scroll = ttk.Scrollbar(bot_container, orient=tk.HORIZONTAL,
                                   command=bot_canvas.xview)
        self.bot_inner = tk.Frame(bot_canvas)
        self.bot_inner.bind(
            "<Configure>",
            lambda _: bot_canvas.configure(
                scrollregion=bot_canvas.bbox("all")))
        bot_canvas.create_window((0, 0), window=self.bot_inner, anchor="nw")
        bot_canvas.configure(xscrollcommand=bot_scroll.set)
        bot_canvas.pack(fill=tk.BOTH, expand=True)
        bot_scroll.pack(fill=tk.X)

        self.lbl_members: list[tk.Label] = []
        for i, label_text in enumerate(self.member_labels_text):
            col = tk.Frame(self.bot_inner, padx=3)
            col.pack(side=tk.LEFT, anchor="n")
            tk.Label(col, text=label_text, font=("Consolas", 7),
                     wraplength=self.MEMBER_W).pack()
            img_lbl = tk.Label(col)
            img_lbl.pack()
            self.lbl_members.append(img_lbl)

        self._img_cache: list[Any] = [None] * (2 + len(self.caps_mem))

        self._show_frame(0)

    def _resize(self, frame: np.ndarray, width: int) -> np.ndarray:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        h, w = frame.shape[:2]
        scale = width / max(w, 1)
        return cv2.resize(frame, (width, max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)

    def _show_frame(self, idx: int) -> None:
        idx = max(0, min(idx, self.total_frames - 1))
        self.current_frame = idx

        self.cap_raw.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret_r, fr_r = self.cap_raw.read()
        if ret_r and fr_r is not None:
            self._img_cache[0] = cv2_to_tk(self._resize(fr_r, self.TOP_W))
            self.lbl_raw.configure(image=self._img_cache[0])

        self.cap_con.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret_c, fr_c = self.cap_con.read()
        if ret_c and fr_c is not None:
            self._img_cache[1] = cv2_to_tk(self._resize(fr_c, self.TOP_W))
            self.lbl_con.configure(image=self._img_cache[1])

        for i, cap in enumerate(self.caps_mem):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret_m, fr_m = cap.read()
            if ret_m and fr_m is not None:
                self._img_cache[2 + i] = cv2_to_tk(
                    self._resize(fr_m, self.MEMBER_W))
                self.lbl_members[i].configure(image=self._img_cache[2 + i])

        self._updating_slider = True
        self.slider.set(idx)
        self._updating_slider = False
        self.frame_label.configure(
            text=f"Frame {idx} / {self.total_frames - 1}")

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.btn_play.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._play_loop()

    def _play_loop(self) -> None:
        if not self.playing:
            return
        nxt = self.current_frame + 1
        if nxt >= self.total_frames:
            self.playing = False
            self.btn_play.configure(text="Play")
            return
        self._show_frame(nxt)
        delay = max(1, int(1000.0 / (self.fps * self.speed)))
        self._after_id = self.win.after(delay, self._play_loop)

    def _step(self, delta: int) -> None:
        self.playing = False
        self.btn_play.configure(text="Play")
        self._show_frame(self.current_frame + delta)

    def _on_slider(self, val: str) -> None:
        if self._updating_slider:
            return
        self._show_frame(int(float(val)))

    def _update_speed(self) -> None:
        self.speed = max(0.1, self.speed_var.get())

    def _on_close(self) -> None:
        self.playing = False
        if self._after_id:
            try:
                self.win.after_cancel(self._after_id)
            except Exception:
                pass
        self.cap_raw.release()
        self.cap_con.release()
        for c in self.caps_mem:
            c.release()
        self.win.destroy()


# ---------------------------------------------------------------------------
# Distributions tab
# ---------------------------------------------------------------------------

class DistributionsTab:
    """Histograms, box plots, CI bars, and per-track distributions."""

    def __init__(self, parent: tk.Frame, runs: list[RunInfo],
                 analytics: EnsembleAnalytics):
        self.runs = runs
        self.analytics = analytics
        self._parent = parent
        self._build_ui()

    def _build_ui(self) -> None:
        ctrl = tk.Frame(self._parent)
        ctrl.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(ctrl, text="Metric:").pack(side=tk.LEFT, padx=2)
        self._metric_var = tk.StringVar(value="tracks_metric")
        cb = ttk.Combobox(ctrl, textvariable=self._metric_var,
                          values=METRIC_CHOICES, width=18, state="readonly")
        cb.pack(side=tk.LEFT, padx=2)
        cb.bind("<<ComboboxSelected>>", lambda _: self.refresh())

        ttk.Label(ctrl, text="Group by:").pack(side=tk.LEFT, padx=(12, 2))
        avail = self.analytics.available_params()
        self._group_var = tk.StringVar(
            value=avail[0] if avail else "bg_threshold")
        gb = ttk.Combobox(ctrl, textvariable=self._group_var,
                          values=avail, width=20, state="readonly")
        gb.pack(side=tk.LEFT, padx=2)
        gb.bind("<<ComboboxSelected>>", lambda _: self.refresh())

        self._sel_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Selected only",
                        variable=self._sel_only_var,
                        command=self.refresh).pack(side=tk.LEFT, padx=12)

        self._summary_lbl = ttk.Label(self._parent, text="",
                                      font=("Consolas", 9), anchor="w",
                                      justify=tk.LEFT)
        self._summary_lbl.pack(fill=tk.X, padx=8, pady=(0, 2))

        self.fig = Figure(figsize=(13, 6.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self._parent)
        toolbar = NavigationToolbar2Tk(self.canvas, self._parent)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4)

    def refresh(self) -> None:
        self._redraw()

    def _active_runs(self) -> list[RunInfo]:
        if self._sel_only_var.get():
            sel = [r for r in self.runs if r.selected]
            return sel if sel else self.runs
        return self.runs

    def _collect_track_array(self, attr: str,
                             max_total: int = 50000) -> np.ndarray:
        """Aggregate per-track data from active runs, subsampled."""
        runs = self._active_runs()
        arrs: list[np.ndarray] = []
        per_run_max = max(100, max_total // max(len(runs), 1))
        rng = np.random.default_rng(0)
        for r in runs:
            ts = r.track_stats
            if not ts._parsed or ts.n_entries == 0:
                continue
            arr = getattr(ts, attr, np.array([]))
            if len(arr) == 0:
                continue
            if len(arr) > per_run_max:
                arr = rng.choice(arr, per_run_max, replace=False)
            arrs.append(arr)
        return np.concatenate(arrs) if arrs else np.array([])

    def _redraw(self) -> None:
        self.fig.clf()
        metric = self._metric_var.get()
        group = self._group_var.get()
        runs = self._active_runs()

        vals = self.analytics.get_metric_array(metric, runs)
        n_sel = sum(1 for r in self.runs if r.selected)
        if len(vals) > 0:
            m, ci_lo, ci_hi = self.analytics.bootstrap_ci(vals)
            std = float(np.std(vals))
            self._summary_lbl.configure(
                text=(f"Showing: {len(runs)}/{len(self.runs)} runs  |  "
                      f"Selected: {n_sel}  |  "
                      f"{metric}: mean={m:.1f} \u00b1 {std:.1f}, "
                      f"median={np.median(vals):.1f}, "
                      f"range=[{vals.min():.1f}, {vals.max():.1f}], "
                      f"95% CI=[{ci_lo:.1f}, {ci_hi:.1f}]"))
        else:
            self._summary_lbl.configure(text="No data")

        # Row 0: per-run metric distributions
        ax1 = self.fig.add_subplot(2, 3, 1)
        self._draw_metric_hist(ax1, vals, metric)

        ax2 = self.fig.add_subplot(2, 3, 2)
        self._draw_boxplot(ax2, group, metric, runs)

        ax3 = self.fig.add_subplot(2, 3, 3)
        self._draw_ci_bars(ax3, group, metric, runs)

        # Row 1: per-track distributions (need tracks loaded)
        ax4 = self.fig.add_subplot(2, 3, 4)
        ax5 = self.fig.add_subplot(2, 3, 5)
        ax6 = self.fig.add_subplot(2, 3, 6)

        if self.analytics._tracks_loaded:
            self._draw_track_dist(ax4, "confidences", "Track Confidence")
            self._draw_track_dist(ax5, "speeds", "Track Speed")
            self._draw_lifetime_dist(ax6)
        else:
            for ax, title in [(ax4, "Track Confidence"),
                              (ax5, "Track Speed"),
                              (ax6, "Track Lifetime")]:
                ax.text(0.5, 0.5,
                        "Track data not loaded\n"
                        "(click Distributions / Explorer tab to trigger)",
                        ha="center", va="center", fontsize=9, color="gray",
                        transform=ax.transAxes)
                ax.set_title(title, fontsize=9)
                ax.set_axis_off()

        self.fig.tight_layout()
        self.canvas.draw()

    # ---- Individual subplot drawers ----

    def _draw_metric_hist(self, ax, vals: np.ndarray, metric: str) -> None:
        if len(vals) == 0:
            ax.set_title(f"{metric} (no data)", fontsize=9)
            return
        bins = min(30, max(5, len(vals) // 3))
        ax.hist(vals, bins=bins, color="steelblue", alpha=0.7, density=True,
                edgecolor="white", linewidth=0.5)
        m = float(np.mean(vals))
        ax.axvline(m, color="red", ls="--", lw=1.2,
                   label=f"\u03bc={m:.1f}")
        _, ci_lo, ci_hi = self.analytics.bootstrap_ci(vals)
        ax.axvspan(ci_lo, ci_hi, alpha=0.15, color="red", label="95% CI")
        if (_gaussian_kde is not None and len(vals) > 3
                and np.std(vals) > 1e-10):
            kde = _gaussian_kde(vals)
            x = np.linspace(float(vals.min()), float(vals.max()), 200)
            ax.plot(x, kde(x), color="darkblue", lw=1.2)
        ax.set_title(f"{metric} distribution", fontsize=9)
        ax.set_xlabel(metric, fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    def _draw_boxplot(self, ax, param: str, metric: str,
                      runs: list[RunInfo]) -> None:
        groups = self.analytics.group_by_param(param, metric, runs)
        if not groups:
            ax.set_title(f"{metric} by {param} (no data)", fontsize=9)
            return
        labels = [f"{k:g}" for k in groups]
        data = list(groups.values())
        ax.boxplot(data, labels=labels, patch_artist=True,
                   boxprops=dict(facecolor="lightsteelblue", alpha=0.7),
                   medianprops=dict(color="red", linewidth=1.5))
        ax.set_title(f"{metric} by {param}", fontsize=9)
        ax.set_xlabel(param, fontsize=8)
        ax.set_ylabel(metric, fontsize=8)
        ax.tick_params(labelsize=7)
        if len(labels) > 6:
            ax.set_xticklabels(labels, rotation=45, ha="right")

    def _draw_ci_bars(self, ax, param: str, metric: str,
                      runs: list[RunInfo]) -> None:
        ci_data = self.analytics.param_level_ci(param, metric, runs)
        if not ci_data:
            ax.set_title(f"CI: {metric} by {param} (no data)", fontsize=9)
            return
        levels, means, los, his = zip(*ci_data)
        x = np.arange(len(levels))
        errs_lo = np.array(means) - np.array(los)
        errs_hi = np.array(his) - np.array(means)
        ax.bar(x, means, color="steelblue", alpha=0.7,
               yerr=[errs_lo, errs_hi], capsize=4,
               error_kw=dict(ecolor="darkred", linewidth=1.2))
        labels = [f"{lv:g}" for lv in levels]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        if len(labels) > 6:
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"Mean {metric} \u00b1 95% CI", fontsize=9)
        ax.set_xlabel(param, fontsize=8)
        ax.set_ylabel(f"Mean {metric}", fontsize=8)
        ax.tick_params(labelsize=7)

    def _draw_track_dist(self, ax, attr: str, title: str) -> None:
        data = self._collect_track_array(attr)
        if len(data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=9, color="gray", transform=ax.transAxes)
            ax.set_title(title, fontsize=9)
            return
        bins = min(50, max(10, len(data) // 100))
        ax.hist(data, bins=bins, color="mediumpurple", alpha=0.7,
                density=True, edgecolor="white", linewidth=0.3)
        m = float(np.mean(data))
        s = float(np.std(data))
        ax.axvline(m, color="red", ls="--", lw=1,
                   label=f"\u03bc={m:.3f} \u00b1 {s:.3f}")
        n = len(data)
        se = s / np.sqrt(n) if n > 1 else 0
        ax.axvspan(m - 1.96 * se, m + 1.96 * se, alpha=0.12, color="red",
                   label="95% CI of mean")
        if (_gaussian_kde is not None and len(data) > 3 and s > 1e-10):
            sample = data if len(data) <= 10000 else \
                np.random.default_rng(0).choice(data, 10000, replace=False)
            try:
                kde = _gaussian_kde(sample)
                xr = np.linspace(float(data.min()), float(data.max()), 200)
                ax.plot(xr, kde(xr), color="indigo", lw=1.2)
            except Exception:
                pass
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)

    def _draw_lifetime_dist(self, ax) -> None:
        data = self._collect_track_array("track_lifetimes")
        if len(data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=9, color="gray", transform=ax.transAxes)
            ax.set_title("Track Lifetime", fontsize=9)
            return
        rng = float(data.max()) - float(data.min()) + 1
        bins = min(50, max(5, int(rng)))
        ax.hist(data, bins=bins, color="seagreen", alpha=0.7,
                edgecolor="white", linewidth=0.3)
        m = float(np.mean(data))
        med = float(np.median(data))
        ax.axvline(m, color="red", ls="--", lw=1,
                   label=f"\u03bc={m:.1f}")
        ax.axvline(med, color="orange", ls=":", lw=1,
                   label=f"median={med:.1f}")
        ax.set_title("Track Lifetime (frames)", fontsize=9)
        ax.set_xlabel("Frames", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Parameter Explorer tab  (plots on left, run list on right)
# ---------------------------------------------------------------------------

_CHECK_ON = "\u2611"
_CHECK_OFF = "\u2610"


class ExplorerTab:
    """Scatter plot, 2-D heatmap, parallel coordinates, and a run-list
    sidebar with checkboxes.  Click any scatter point to open the
    side-by-side comparison video.  Use the checkboxes in the run list
    (or the Grid tab) to mark runs for consensus."""

    def __init__(self, parent: tk.Frame, runs: list[RunInfo],
                 analytics: EnsembleAnalytics,
                 on_selection_changed, on_open_compare):
        self.runs = runs
        self.analytics = analytics
        self._on_selection_changed = on_selection_changed
        self._on_open_compare = on_open_compare
        self._parent = parent
        self._ax_scatter = None
        self._scatter_x: np.ndarray = np.array([])
        self._scatter_y: np.ndarray = np.array([])
        self._cid_click: int | None = None
        self._cid_motion: int | None = None
        self._hover_annot = None
        self._updating_tree = False
        self._build_ui()

    def _build_ui(self) -> None:
        # ---- Control bar ----
        ctrl = tk.Frame(self._parent)
        ctrl.pack(fill=tk.X, padx=4, pady=4)

        avail = self.analytics.available_params()

        ttk.Label(ctrl, text="X axis:").pack(side=tk.LEFT, padx=2)
        self._x_var = tk.StringVar(
            value=avail[0] if avail else "bg_threshold")
        xc = ttk.Combobox(ctrl, textvariable=self._x_var, values=avail,
                          width=18, state="readonly")
        xc.pack(side=tk.LEFT, padx=2)
        xc.bind("<<ComboboxSelected>>", lambda _: self.refresh())

        ttk.Label(ctrl, text="Y axis:").pack(side=tk.LEFT, padx=(8, 2))
        self._y_var = tk.StringVar(
            value=avail[1] if len(avail) > 1 else
                  (avail[0] if avail else "min_intensity"))
        yc = ttk.Combobox(ctrl, textvariable=self._y_var, values=avail,
                          width=18, state="readonly")
        yc.pack(side=tk.LEFT, padx=2)
        yc.bind("<<ComboboxSelected>>", lambda _: self.refresh())

        ttk.Label(ctrl, text="Color:").pack(side=tk.LEFT, padx=(8, 2))
        self._color_var = tk.StringVar(value="tracks_metric")
        cc = ttk.Combobox(ctrl, textvariable=self._color_var,
                          values=METRIC_CHOICES, width=18, state="readonly")
        cc.pack(side=tk.LEFT, padx=2)
        cc.bind("<<ComboboxSelected>>", lambda _: self.refresh())

        ttk.Label(ctrl, text="Size:").pack(side=tk.LEFT, padx=(8, 2))
        self._size_var = tk.StringVar(value="total_detections")
        sc = ttk.Combobox(ctrl, textvariable=self._size_var,
                          values=METRIC_CHOICES, width=18, state="readonly")
        sc.pack(side=tk.LEFT, padx=2)
        sc.bind("<<ComboboxSelected>>", lambda _: self.refresh())

        # ---- PanedWindow: plots (left) | run list (right) ----
        self._pane = ttk.PanedWindow(self._parent, orient=tk.HORIZONTAL)
        self._pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        # Left side — matplotlib plots
        plot_frame = tk.Frame(self._pane)
        self._pane.add(plot_frame, weight=3)

        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._cid_click = self.canvas.mpl_connect(
            "button_press_event", self._on_scatter_click)
        self._cid_motion = self.canvas.mpl_connect(
            "motion_notify_event", self._on_hover)

        # Right side — run list with checkboxes
        list_frame = tk.Frame(self._pane, width=320)
        self._pane.add(list_frame, weight=1)

        list_header = tk.Frame(list_frame)
        list_header.pack(fill=tk.X, padx=2, pady=(4, 2))
        tk.Label(list_header, text="Run List",
                 font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=4)
        ttk.Button(list_header, text="Select All",
                   command=self._list_select_all).pack(side=tk.RIGHT, padx=2)
        ttk.Button(list_header, text="Clear All",
                   command=self._list_clear_all).pack(side=tk.RIGHT, padx=2)

        tk.Label(list_frame,
                 text="Click row \u2192 open video  |  "
                      "Click \u2610 \u2192 toggle selection",
                 font=("Segoe UI", 7), fg="gray").pack(
                     anchor="w", padx=6, pady=(0, 2))

        cols = ("sel", "label", "tracks", "conf", "speed")
        self._tree = ttk.Treeview(
            list_frame, columns=cols, show="headings",
            selectmode="browse", height=20)
        self._tree.heading("sel", text="\u2610")
        self._tree.heading("label", text="Run")
        self._tree.heading("tracks", text="Tracks")
        self._tree.heading("conf", text="Conf")
        self._tree.heading("speed", text="Speed")
        self._tree.column("sel", width=30, anchor="center", stretch=False)
        self._tree.column("label", width=140, anchor="w")
        self._tree.column("tracks", width=55, anchor="e")
        self._tree.column("conf", width=55, anchor="e")
        self._tree.column("speed", width=55, anchor="e")

        tree_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                    command=self._tree.yview)
        self._tree.configure(yscrollcommand=tree_scroll.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._tree.tag_configure("selected_row",
                                 background="#d4edda")

        self._tree.bind("<ButtonRelease-1>", self._on_tree_click)

        self._populate_tree()

    # ---- Tree helpers ----

    def _populate_tree(self) -> None:
        self._updating_tree = True
        self._tree.delete(*self._tree.get_children())
        for run in self.runs:
            ts = run.track_stats
            conf_str = (f"{ts.confidence_mean:.3f}"
                        if ts._parsed and ts.n_entries > 0 else "")
            spd_str = (f"{ts.speed_mean:.2f}"
                       if ts._parsed and ts.n_entries > 0 else "")
            check = _CHECK_ON if run.selected else _CHECK_OFF
            tags = ("selected_row",) if run.selected else ()
            self._tree.insert(
                "", tk.END, iid=run.key,
                values=(check, run.label, run.tracks_metric,
                        conf_str, spd_str),
                tags=tags)
        self._updating_tree = False

    def _sync_tree_selection(self) -> None:
        """Update checkmarks and row highlights to match run.selected."""
        self._updating_tree = True
        for run in self.runs:
            check = _CHECK_ON if run.selected else _CHECK_OFF
            tags = ("selected_row",) if run.selected else ()
            try:
                self._tree.item(run.key, values=(
                    check, run.label, run.tracks_metric,
                    f"{run.track_stats.confidence_mean:.3f}"
                    if run.track_stats._parsed and
                    run.track_stats.n_entries > 0 else "",
                    f"{run.track_stats.speed_mean:.2f}"
                    if run.track_stats._parsed and
                    run.track_stats.n_entries > 0 else "",
                ), tags=tags)
            except Exception:
                pass
        self._updating_tree = False

    def _on_tree_click(self, event) -> None:
        if self._updating_tree:
            return
        row_id = self._tree.identify_row(event.y)
        col = self._tree.identify_column(event.x)
        if not row_id:
            return
        run = next((r for r in self.runs if r.key == row_id), None)
        if run is None:
            return

        if col == "#1":
            run.selected = not run.selected
            self._sync_tree_selection()
            self._on_selection_changed()
        else:
            self._on_open_compare(run)

    def _list_select_all(self) -> None:
        for r in self.runs:
            r.selected = True
        self._sync_tree_selection()
        self._on_selection_changed()

    def _list_clear_all(self) -> None:
        for r in self.runs:
            r.selected = False
        self._sync_tree_selection()
        self._on_selection_changed()

    # ---- Refresh / redraw ----

    def refresh(self) -> None:
        self._sync_tree_selection()
        self._redraw()

    def _redraw(self) -> None:
        if self._cid_click is not None:
            self.canvas.mpl_disconnect(self._cid_click)
            self._cid_click = None
        if self._cid_motion is not None:
            self.canvas.mpl_disconnect(self._cid_motion)
            self._cid_motion = None

        self.fig.clf()
        gs = self.fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

        self._ax_scatter = self.fig.add_subplot(gs[0, :2])
        self._draw_scatter(self._ax_scatter)

        ax_heat = self.fig.add_subplot(gs[0, 2])
        self._draw_heatmap(ax_heat)

        ax_par = self.fig.add_subplot(gs[1, :])
        self._draw_parallel_coords(ax_par)

        self.fig.tight_layout()
        self.canvas.draw()

        self._cid_click = self.canvas.mpl_connect(
            "button_press_event", self._on_scatter_click)
        self._cid_motion = self.canvas.mpl_connect(
            "motion_notify_event", self._on_hover)

    # ---- Scatter plot ----

    def _draw_scatter(self, ax) -> None:
        x_param = self._x_var.get()
        y_param = self._y_var.get()
        color_metric = self._color_var.get()
        size_metric = self._size_var.get()

        self._scatter_x = np.array(
            [r.get_metric(x_param) for r in self.runs])
        self._scatter_y = np.array(
            [r.get_metric(y_param) for r in self.runs])
        colors = np.array(
            [r.get_metric(color_metric) for r in self.runs])
        sizes = np.array(
            [r.get_metric(size_metric) for r in self.runs])

        smin, smax = float(sizes.min()), float(sizes.max())
        if smax > smin:
            norm_sizes = 30 + 200 * (sizes - smin) / (smax - smin)
        else:
            norm_sizes = np.full_like(sizes, 80.0)

        cmin, cmax = float(colors.min()), float(colors.max())
        sel_mask = np.array([r.selected for r in self.runs])
        unsel = ~sel_mask

        if np.any(unsel):
            ax.scatter(self._scatter_x[unsel], self._scatter_y[unsel],
                       c=colors[unsel], cmap="RdYlGn",
                       s=norm_sizes[unsel], alpha=0.6,
                       edgecolors="gray", linewidths=0.5,
                       vmin=cmin, vmax=cmax)
        if np.any(sel_mask):
            ax.scatter(self._scatter_x[sel_mask], self._scatter_y[sel_mask],
                       c=colors[sel_mask], cmap="RdYlGn",
                       s=norm_sizes[sel_mask] * 1.5, alpha=1.0,
                       edgecolors="black", linewidths=2, marker="*",
                       vmin=cmin, vmax=cmax)

        norm = Normalize(vmin=cmin, vmax=cmax)
        sm = mpl_cm.ScalarMappable(cmap=mpl_cm.RdYlGn, norm=norm)
        sm.set_array([])
        self.fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04,
                          label=color_metric)

        ax.set_xlabel(x_param, fontsize=9)
        ax.set_ylabel(y_param, fontsize=9)
        ax.set_title(
            f"Click a point to open side-by-side comparison video  "
            f"(size = {size_metric})", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

        self._hover_annot = ax.annotate(
            "", xy=(0, 0), xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="#ffffe0",
                      ec="gray", alpha=0.95),
            fontsize=7, family="monospace",
            arrowprops=dict(arrowstyle="->", color="gray"),
            visible=False)

    # ---- 2-D heatmap ----

    def _draw_heatmap(self, ax) -> None:
        x_param = self._x_var.get()
        y_param = self._y_var.get()
        color_metric = self._color_var.get()

        if x_param == y_param:
            ax.text(0.5, 0.5, "Select different\nX and Y axes",
                    ha="center", va="center", fontsize=10, color="gray",
                    transform=ax.transAxes)
            ax.set_title("2D Heatmap", fontsize=10)
            return

        x_vals = sorted({float(r.overrides.get(x_param, 0))
                         for r in self.runs
                         if r.overrides.get(x_param) is not None})
        y_vals = sorted({float(r.overrides.get(y_param, 0))
                         for r in self.runs
                         if r.overrides.get(y_param) is not None})
        if not x_vals or not y_vals:
            ax.text(0.5, 0.5, "Insufficient data",
                    ha="center", va="center", fontsize=10, color="gray",
                    transform=ax.transAxes)
            ax.set_title("2D Heatmap", fontsize=10)
            return

        cells: dict[tuple[float, float], list[float]] = defaultdict(list)
        for r in self.runs:
            xv = r.overrides.get(x_param)
            yv = r.overrides.get(y_param)
            if xv is None or yv is None:
                continue
            cells[(float(xv), float(yv))].append(r.get_metric(color_metric))

        grid = np.full((len(y_vals), len(x_vals)), np.nan)
        for (xv, yv), vals in cells.items():
            xi = x_vals.index(xv)
            yi = y_vals.index(yv)
            grid[yi, xi] = float(np.mean(vals))

        masked = np.ma.array(grid, mask=np.isnan(grid))
        im = ax.imshow(masked, cmap="RdYlGn", aspect="auto",
                       origin="lower", interpolation="nearest")
        self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([f"{v:g}" for v in x_vals], fontsize=6,
                           rotation=45, ha="right")
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([f"{v:g}" for v in y_vals], fontsize=6)
        ax.set_xlabel(x_param, fontsize=8)
        ax.set_ylabel(y_param, fontsize=8)
        ax.set_title(f"Mean {color_metric}", fontsize=9)

        fsize = max(5, min(8, 80 // max(len(x_vals), len(y_vals))))
        for yi in range(len(y_vals)):
            for xi in range(len(x_vals)):
                val = grid[yi, xi]
                if not np.isnan(val):
                    ax.text(xi, yi, f"{val:.0f}", ha="center", va="center",
                            fontsize=fsize, color="black")

    # ---- Parallel coordinates ----

    def _draw_parallel_coords(self, ax) -> None:
        params = self.analytics.available_params()
        color_metric = self._color_var.get()

        if not params:
            ax.text(0.5, 0.5, "No varying parameters",
                    ha="center", va="center", fontsize=10, color="gray",
                    transform=ax.transAxes)
            ax.set_title("Parallel Coordinates", fontsize=10)
            return

        axes_labels = params + [color_metric]
        axes_data: list[np.ndarray] = []
        for p in params:
            axes_data.append(
                np.array([r.get_metric(p) for r in self.runs]))
        metric_vals = self.analytics.get_metric_array(color_metric)
        axes_data.append(metric_vals)

        n_axes = len(axes_data)
        x_pos = np.arange(n_axes)

        normed: list[np.ndarray] = []
        axes_min: list[float] = []
        axes_max: list[float] = []
        for vals in axes_data:
            vmin, vmax = float(vals.min()), float(vals.max())
            axes_min.append(vmin)
            axes_max.append(vmax)
            if vmax > vmin:
                normed.append((vals - vmin) / (vmax - vmin))
            else:
                normed.append(np.full_like(vals, 0.5, dtype=float))

        for i in range(n_axes):
            ax.axvline(i, color="gray", lw=0.5, alpha=0.5)

        norm = Normalize(vmin=float(metric_vals.min()),
                         vmax=float(metric_vals.max()))
        cmap = mpl_cm.RdYlGn

        for selected_pass in (False, True):
            for j in range(len(self.runs)):
                if self.runs[j].selected != selected_pass:
                    continue
                y = [float(normed[i][j]) for i in range(n_axes)]
                color = cmap(norm(metric_vals[j]))
                alpha = 0.85 if selected_pass else 0.15
                lw = 2.5 if selected_pass else 0.7
                ax.plot(x_pos, y, color=color, alpha=alpha, lw=lw)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(axes_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(-0.08, 1.12)
        ax.set_ylabel("Normalized value", fontsize=8)
        ax.set_title(
            f"Parallel Coordinates  (color = {color_metric})", fontsize=10)
        ax.tick_params(axis="y", labelsize=7)

        for i in range(n_axes):
            ax.text(i, -0.06, f"{axes_min[i]:g}",
                    ha="center", fontsize=6, color="gray")
            ax.text(i, 1.06, f"{axes_max[i]:g}",
                    ha="center", fontsize=6, color="gray")

    # ---- Scatter interaction: click to compare, hover tooltip ----

    def _find_nearest(self, event) -> int | None:
        """Return index of nearest scatter point, or None."""
        if (self._ax_scatter is None
                or event.inaxes != self._ax_scatter
                or len(self._scatter_x) == 0
                or event.xdata is None):
            return None
        pts = np.column_stack([self._scatter_x, self._scatter_y])
        click = np.array([[event.xdata, event.ydata]])
        pts_disp = self._ax_scatter.transData.transform(pts)
        click_disp = self._ax_scatter.transData.transform(click)[0]
        dists = np.sqrt(np.sum((pts_disp - click_disp) ** 2, axis=1))
        idx = int(np.argmin(dists))
        return idx if dists[idx] <= 30 else None

    def _on_scatter_click(self, event) -> None:
        idx = self._find_nearest(event)
        if idx is None:
            return
        run = self.runs[idx]
        self._on_open_compare(run)

    def _on_hover(self, event) -> None:
        if self._hover_annot is None:
            return
        idx = self._find_nearest(event)
        if idx is None:
            self._hover_annot.set_visible(False)
            self.canvas.draw_idle()
            return
        run = self.runs[idx]
        x = self._scatter_x[idx]
        y = self._scatter_y[idx]
        self._hover_annot.xy = (x, y)
        sel_mark = " [SELECTED]" if run.selected else ""
        ts = run.track_stats
        conf_str = (f"  conf={ts.confidence_mean:.3f}"
                    if ts._parsed and ts.n_entries > 0 else "")
        text = (f"{run.label}{sel_mark}\n"
                f"  tracks={run.tracks_metric}"
                f"  det={run.total_detections}{conf_str}\n"
                f"  (click to open video)")
        self._hover_annot.set_text(text)
        self._hover_annot.set_visible(True)
        self.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Main selector app (tabbed layout)
# ---------------------------------------------------------------------------

class SelectorApp:
    GRID_COLS = 5

    def __init__(self, run_dir: Path, input_video: Path | None):
        self.run_dir = run_dir
        self.input_video = input_video
        self.runs = discover_runs(run_dir)
        if not self.runs:
            print(f"No tracker runs found in {run_dir}", file=sys.stderr)
            sys.exit(1)

        self.analytics = EnsembleAnalytics(self.runs)

        self.root = tk.Tk()
        self.root.title(
            f"Consensus Selector \u2014 {run_dir.name} "
            f"({len(self.runs)} runs)")
        self.root.geometry("1400x850")

        self._thumb_cache: dict[str, ImageTk.PhotoImage] = {}
        self._raw_thumbs: dict[str, np.ndarray] = {}
        self._loading_tracks = False
        self._tooltips: list[ToolTip] = []

        self._build_ui()
        self._load_thumbnails()

    # ---- UI construction ----

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=4)

        ttk.Button(top, text="Select All",
                   command=self._select_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Deselect All",
                   command=self._deselect_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Build Consensus",
                   command=self._build_consensus).pack(side=tk.RIGHT, padx=4)
        self.status_var = tk.StringVar(value="0 selected")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.RIGHT,
                                                          padx=12)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        grid_frame = tk.Frame(self.notebook)
        self.notebook.add(grid_frame, text="Grid")
        self._build_grid_tab(grid_frame)

        dist_frame = tk.Frame(self.notebook)
        self.notebook.add(dist_frame, text="Distributions")
        self.dist_tab = DistributionsTab(dist_frame, self.runs,
                                         self.analytics)

        explorer_frame = tk.Frame(self.notebook)
        self.notebook.add(explorer_frame, text="Explorer")
        self.explorer_tab = ExplorerTab(
            explorer_frame, self.runs, self.analytics,
            on_selection_changed=self._on_selection_changed,
            on_open_compare=self._open_compare,
        )

    def _build_grid_tab(self, parent: tk.Frame) -> None:
        ctrl = tk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(ctrl, text="Sort by:").pack(side=tk.LEFT, padx=2)
        self._sort_var = tk.StringVar(value="tracks_metric")
        sort_cb = ttk.Combobox(ctrl, textvariable=self._sort_var,
                               values=ALL_SORT_OPTIONS, width=18,
                               state="readonly")
        sort_cb.pack(side=tk.LEFT, padx=2)
        sort_cb.bind("<<ComboboxSelected>>",
                     lambda _: self._on_sort_changed())

        self._sort_desc_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="\u2193 Desc",
                        variable=self._sort_desc_var,
                        command=self._on_sort_changed
                        ).pack(side=tk.LEFT, padx=2)

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        ttk.Label(ctrl, text="Color by:").pack(side=tk.LEFT, padx=2)
        self._color_var = tk.StringVar(value="none")
        color_cb = ttk.Combobox(ctrl, textvariable=self._color_var,
                                values=["none"] + ALL_SORT_OPTIONS, width=18,
                                state="readonly")
        color_cb.pack(side=tk.LEFT, padx=2)
        color_cb.bind("<<ComboboxSelected>>",
                      lambda _: self._on_color_changed())

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        ttk.Label(ctrl, text="Auto-select top:").pack(
            side=tk.LEFT, padx=2)
        self._topn_var = tk.IntVar(value=10)
        ttk.Spinbox(ctrl, from_=1, to=max(1, len(self.runs)),
                    textvariable=self._topn_var,
                    width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="Top N",
                   command=self._auto_select_top).pack(
                       side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="Bottom N",
                   command=self._auto_select_bottom).pack(
                       side=tk.LEFT, padx=2)

        container = tk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL,
                                  command=self.canvas.yview)
        self.scrollable = tk.Frame(self.canvas)
        self.scrollable.bind(
            "<Configure>",
            lambda _: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.canvas.yview_scroll(
                -1 * (e.delta // 120), "units"))

        self.cell_frames: dict[str, tk.Frame] = {}
        self.cell_labels: dict[str, tk.Label] = {}
        self.cell_text_labels: dict[str, tk.Label] = {}

    # ---- Thumbnail loading ----

    def _load_thumbnails(self) -> None:
        placeholder = np.full((THUMB_SIZE, THUMB_SIZE, 3), 40, dtype=np.uint8)
        cv2.putText(placeholder, "loading...", (20, THUMB_SIZE // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
                    cv2.LINE_AA)

        for idx, run in enumerate(self.runs):
            row, col = divmod(idx, self.GRID_COLS)
            cell = tk.Frame(self.scrollable, bd=2, relief=tk.FLAT,
                            padx=3, pady=3)
            cell.grid(row=row, column=col, padx=4, pady=4)

            ph_img = cv2_to_tk(placeholder)
            self._thumb_cache[f"_ph_{run.key}"] = ph_img

            img_label = tk.Label(cell, image=ph_img, cursor="hand2")
            img_label.pack()
            img_label.bind("<Button-1>",
                           lambda e, r=run: self._open_compare(r))

            txt = tk.Label(cell, text=run.label, font=("Consolas", 8),
                           wraplength=THUMB_SIZE + 10)
            txt.pack()
            txt.bind("<Button-1>",
                     lambda e, r=run: self._open_compare(r))

            self.cell_frames[run.key] = cell
            self.cell_labels[run.key] = img_label
            self.cell_text_labels[run.key] = txt

            self._tooltips.append(
                ToolTip(img_label,
                        lambda r=run: self._tooltip_text(r)))

        self.root.after(50, self._load_thumbs_async)

    def _tooltip_text(self, run: RunInfo) -> str:
        lines: list[str] = [f"Key: {run.key}"]
        if run.tracker_name:
            lines.append(f"Tracker: {run.tracker_name}")
        lines.append(f"Tracks: {run.tracks_metric}")
        lines.append(f"Detections: {run.total_detections}")
        lines.append(f"Runtime: {run.runtime_s:.1f}s")
        ts = run.track_stats
        if ts._parsed and ts.n_entries > 0:
            lines.append(
                f"Mean conf: {ts.confidence_mean:.4f}"
                f" \u00b1 {ts.confidence_std:.4f}")
            lines.append(
                f"Mean speed: {ts.speed_mean:.2f}"
                f" \u00b1 {ts.speed_std:.2f}")
            lines.append(f"Mean lifetime: {ts.lifetime_mean:.1f} frames")
        lines.append("---")
        for k, v in sorted(run.overrides.items()):
            lines.append(f"{k}: {v}")
        return "\n".join(lines)

    def _load_thumbs_async(self) -> None:
        def worker():
            for run in self.runs:
                if not run.has_video:
                    continue
                thumb = extract_thumbnail(run.video_path)
                if thumb is not None:
                    self._raw_thumbs[run.key] = thumb
                    self.root.after(0, self._update_thumb_display, run.key)

        threading.Thread(target=worker, daemon=True).start()

    def _update_thumb_display(self, key: str) -> None:
        if key not in self._raw_thumbs or key not in self.cell_labels:
            return
        run = next((r for r in self.runs if r.key == key), None)
        if run is None:
            return

        thumb = self._raw_thumbs[key].copy()

        color_metric = self._color_var.get()
        if color_metric != "none":
            vals = self.analytics.get_metric_array(color_metric)
            if len(vals) > 0:
                vmin, vmax = float(vals.min()), float(vals.max())
                bgr = metric_color_bgr(run.get_metric(color_metric),
                                       vmin, vmax)
                thumb[-8:, :] = bgr

        if run.selected:
            cv2.rectangle(thumb, (0, 0),
                          (thumb.shape[1] - 1, thumb.shape[0] - 1),
                          (0, 200, 0), 4)

        img = cv2_to_tk(thumb)
        self._thumb_cache[key] = img
        self.cell_labels[key].configure(image=img)

    def _refresh_all_thumbs(self) -> None:
        for key in self._raw_thumbs:
            self._update_thumb_display(key)

    # ---- Sort / color / auto-select ----

    def _on_sort_changed(self) -> None:
        metric = self._sort_var.get()
        if metric in TRACK_DEPENDENT and not self.analytics._tracks_loaded:
            self._ensure_tracks_loaded(self._sort_grid)
        else:
            self._sort_grid()

    def _sort_grid(self) -> None:
        metric = self._sort_var.get()
        desc = self._sort_desc_var.get()
        self.runs.sort(key=lambda r: r.get_metric(metric), reverse=desc)
        for cell in self.cell_frames.values():
            cell.grid_forget()
        for idx, run in enumerate(self.runs):
            row, col = divmod(idx, self.GRID_COLS)
            self.cell_frames[run.key].grid(row=row, column=col,
                                           padx=4, pady=4)

    def _on_color_changed(self) -> None:
        metric = self._color_var.get()
        if metric in TRACK_DEPENDENT and not self.analytics._tracks_loaded:
            self._ensure_tracks_loaded(self._refresh_all_thumbs)
        else:
            self._refresh_all_thumbs()

    def _auto_select_top(self) -> None:
        self._auto_select(top=True)

    def _auto_select_bottom(self) -> None:
        self._auto_select(top=False)

    def _auto_select(self, top: bool) -> None:
        metric = self._sort_var.get()
        n = self._topn_var.get()
        if metric in TRACK_DEPENDENT and not self.analytics._tracks_loaded:
            self._ensure_tracks_loaded(lambda: self._auto_select(top))
            return
        ranked = sorted(self.runs,
                        key=lambda r: r.get_metric(metric), reverse=top)
        for r in self.runs:
            r.selected = False
        for r in ranked[:n]:
            r.selected = True
        self._on_selection_changed()

    # ---- Selection logic ----

    def _on_selection_changed(self) -> None:
        self._refresh_all_thumbs()
        self._refresh_status()
        try:
            tab = self.notebook.tab(self.notebook.select(), "text")
        except Exception:
            tab = ""
        if tab == "Distributions":
            self.dist_tab.refresh()
        elif tab == "Explorer":
            self.explorer_tab.refresh()

    def _select_all(self) -> None:
        for r in self.runs:
            r.selected = True
        self._on_selection_changed()

    def _deselect_all(self) -> None:
        for r in self.runs:
            r.selected = False
        self._on_selection_changed()

    def _refresh_status(self) -> None:
        n = sum(1 for r in self.runs if r.selected)
        self.status_var.set(f"{n} of {len(self.runs)} selected")

    # ---- Tab change handler ----

    def _on_tab_changed(self, event=None) -> None:
        try:
            tab = self.notebook.tab(self.notebook.select(), "text")
        except Exception:
            return
        if tab in ("Distributions", "Explorer"):
            if (not self.analytics._tracks_loaded
                    and not self._loading_tracks):
                self._ensure_tracks_loaded(self._refresh_active_tab)
            else:
                self._refresh_active_tab()

    def _refresh_active_tab(self) -> None:
        try:
            tab = self.notebook.tab(self.notebook.select(), "text")
        except Exception:
            return
        if tab == "Distributions":
            self.dist_tab.refresh()
        elif tab == "Explorer":
            self.explorer_tab.refresh()

    def _ensure_tracks_loaded(self, callback) -> None:
        if self.analytics._tracks_loaded:
            callback()
            return
        if self._loading_tracks:
            return
        self._loading_tracks = True

        pw = tk.Toplevel(self.root)
        pw.title("Loading track data")
        pw.geometry("360x100")
        pw.transient(self.root)
        pw.grab_set()

        status = ttk.Label(pw, text="Parsing tracks.txt files...")
        status.pack(padx=20, pady=(15, 5))
        pbar = ttk.Progressbar(pw, length=310, mode="determinate",
                               maximum=max(1, len(self.runs)))
        pbar.pack(padx=20, pady=5)

        def worker():
            def prog(done, total):
                self.root.after(0, lambda d=done, t=total: (
                    pbar.configure(value=d),
                    status.configure(text=f"Parsed {d}/{t}"),
                ))
            self.analytics.load_all_tracks(progress_cb=prog)
            self.root.after(0, _done)

        def _done():
            self._loading_tracks = False
            try:
                pw.grab_release()
                pw.destroy()
            except Exception:
                pass
            callback()

        threading.Thread(target=worker, daemon=True).start()

    # ---- Compare window ----

    def _open_compare(self, run: RunInfo) -> None:
        if not run.has_video:
            messagebox.showwarning("No video",
                                   f"No output_video.mp4 for {run.key}")
            return
        if self.input_video is None:
            messagebox.showwarning(
                "No raw video",
                "Cannot find the raw input video. Pass --input-video.")
            return
        CompareWindow(
            self.root, run, self.input_video,
            on_toggle=lambda: (
                self._update_thumb_display(run.key),
                self._refresh_status(),
            ))

    # ---- Consensus building ----

    def _build_consensus(self) -> None:
        selected = [r for r in self.runs if r.selected and r.has_tracks]
        if len(selected) < 2:
            messagebox.showwarning("Too few",
                                   "Select at least 2 runs for consensus.")
            return
        if self.input_video is None:
            messagebox.showwarning(
                "No raw video",
                "Cannot find the raw input video. Pass --input-video.")
            return

        n = len(selected)
        quorum = max(2, math.ceil(n / 3))
        answer = messagebox.askyesno(
            "Build consensus",
            f"Build consensus from {n} selected runs?\n"
            f"Quorum: {quorum}/{n} (at least {quorum} must agree)\n\n"
            f"This may take a minute.",
        )
        if not answer:
            return

        self.status_var.set(f"Building consensus from {n} runs...")
        self.root.update_idletasks()

        def worker():
            try:
                out_path = self._run_consensus(selected, quorum)
                self.root.after(
                    0, lambda: self._consensus_done(out_path, selected))
            except Exception as exc:
                self.root.after(
                    0, lambda: messagebox.showerror("Error", str(exc)))
                self.root.after(0, self._refresh_status)

        threading.Thread(target=worker, daemon=True).start()

    def _run_consensus(self, selected: list[RunInfo],
                       quorum: int) -> Path:
        workspace = Path(__file__).resolve().parent
        if str(workspace) not in sys.path:
            sys.path.insert(0, str(workspace))

        from run_ensemble import (
            build_per_frame_consensus,
            link_consensus_across_frames,
            parse_pmb_track_entries,
        )
        from txtPMB_ensemble import (
            render_consensus_video_smooth,
            smooth_consensus_tracks,
        )

        entries: list[tuple[int, list[tuple[int, int, float, float]]]] = []
        for run in selected:
            idx = len(entries)
            entries.append((idx, parse_pmb_track_entries(run.track_log)))

        n_ok = len(entries)
        eff_quorum = min(quorum, n_ok)

        renumbered = [(i, e) for i, (_s, e) in enumerate(entries)]
        consensus = build_per_frame_consensus(
            renumbered,
            spatial_threshold=6.0,
            quorum=eff_quorum,
            n_trackers=n_ok,
        )
        linked = link_consensus_across_frames(
            consensus, link_threshold=12.0, max_frame_gap=8,
        )
        linked = smooth_consensus_tracks(linked, alpha=0.45)

        out_path = self.run_dir / "consensus_selected.mp4"
        render_consensus_video_smooth(
            self.input_video, out_path, linked, n_ok,
            max_tail=15, fade_frames=6,
        )

        keys = [r.key for r in selected]
        manifest = {
            "selected_runs": keys,
            "n_selected": n_ok,
            "quorum": eff_quorum,
            "consensus_tracks": len(linked),
            "output": str(out_path),
        }
        (self.run_dir / "consensus_selected_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )
        return out_path

    def _consensus_done(self, consensus_path: Path,
                        selected: list[RunInfo]) -> None:
        self._refresh_status()
        members = [
            (r.label, r.video_path) for r in selected if r.has_video
        ]
        ResultsWindow(
            self.root,
            self.input_video,
            consensus_path,
            members,
        )

    # ---- Run ----

    def run(self) -> None:
        self._refresh_status()
        self.root.mainloop()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive consensus model selector with analytics "
                    "dashboard \u2014 browse, compare, and pick runs.",
    )
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Path to a completed sweep output folder")
    p.add_argument("--input-video", type=Path, default=None,
                   help="Raw input video (auto-detected from manifest "
                        "if omitted)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 1

    input_video = resolve_input_video(run_dir, args.input_video)
    if input_video is None:
        print("Warning: could not auto-detect raw input video. "
              "Pass --input-video.", file=sys.stderr)

    app = SelectorApp(run_dir, input_video)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
