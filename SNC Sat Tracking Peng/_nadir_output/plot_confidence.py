"""Plot per-track confidence over frames 0-240."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log_path = Path(__file__).resolve().parent / "nadir_tracks.txt"
out_path = Path(__file__).resolve().parent / "confidence_plot.png"

tracks: dict[int, dict] = {}

with open(log_path) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        frame = int(parts[0])
        tid = int(parts[1])
        conf = float(parts[5])
        x = int(parts[2])
        y = int(parts[3])
        speed = float(parts[6])
        if tid not in tracks:
            tracks[tid] = {"frames": [], "conf": [], "x": [], "y": [], "speed": []}
        tracks[tid]["frames"].append(frame)
        tracks[tid]["conf"].append(conf * 100.0)
        tracks[tid]["x"].append(x)
        tracks[tid]["y"].append(y)
        tracks[tid]["speed"].append(speed)

total_frames = 240
colors = plt.cm.tab10.colors

fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 2, 1]})

# --- Panel 1: Confidence vs Frame ---
ax1 = axes[0]
for i, (tid, data) in enumerate(sorted(tracks.items())):
    c = colors[i % len(colors)]
    f0, f1 = data["frames"][0], data["frames"][-1]
    ax1.plot(data["frames"], data["conf"], "o-", color=c, markersize=2, linewidth=1.5,
             label=f"Track {tid} (frames {f0}-{f1})")

# shade the gaps
all_tracked_frames = set()
for d in tracks.values():
    all_tracked_frames.update(d["frames"])
gap_start = None
for f in range(total_frames):
    if f not in all_tracked_frames:
        if gap_start is None:
            gap_start = f
    else:
        if gap_start is not None:
            ax1.axvspan(gap_start, f, color="red", alpha=0.08)
            gap_start = None
if gap_start is not None:
    ax1.axvspan(gap_start, total_frames, color="red", alpha=0.08)

ax1.set_xlim(0, total_frames)
ax1.set_ylim(0, 105)
ax1.set_ylabel("Confidence (%)")
ax1.set_title("Nadir PMB Tracker — Track Confidence Over Time\n"
              "(red shading = no active track; same object keeps getting new IDs)")
ax1.legend(loc="lower right", fontsize=8)
ax1.grid(True, alpha=0.3)

# --- Panel 2: X position vs Frame (shows it's one object) ---
ax2 = axes[1]
for i, (tid, data) in enumerate(sorted(tracks.items())):
    c = colors[i % len(colors)]
    ax2.plot(data["frames"], data["x"], "o-", color=c, markersize=2, linewidth=1.5,
             label=f"Track {tid}")
# extrapolate track 3 velocity to show where it should have been
t3 = tracks.get(3)
if t3:
    vx = np.mean(np.diff(t3["x"]) / np.diff(t3["frames"]))
    extrap_frames = np.arange(0, total_frames)
    x0 = t3["x"][0] - vx * (t3["frames"][0] - 0)
    extrap_x = x0 + vx * extrap_frames
    ax2.plot(extrap_frames, extrap_x, "--", color="gray", linewidth=0.8, alpha=0.6,
             label=f"Expected trajectory (vx={vx:.1f} px/fr)")

ax2.set_xlim(0, total_frames)
ax2.set_ylabel("X position (px)")
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.3)

# --- Panel 3: Per-frame detection count from log ---
ax3 = axes[2]
det_per_frame = np.zeros(total_frames, dtype=int)
for d in tracks.values():
    for f in d["frames"]:
        det_per_frame[f] += 1
ax3.bar(range(total_frames), det_per_frame, width=1.0, color="steelblue", alpha=0.7)
ax3.set_xlim(0, total_frames)
ax3.set_xlabel("Frame")
ax3.set_ylabel("Confirmed tracks")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(out_path), dpi=150)
print(f"Saved: {out_path}")
