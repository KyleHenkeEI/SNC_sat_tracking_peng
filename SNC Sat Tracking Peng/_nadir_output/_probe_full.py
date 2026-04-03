"""Full-video diagnostic: what does each residual frame look like, where are the targets."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from nadir_tracking.ego_motion import motion_compensated_residual

video = str(ROOT / "synthetic_LEO_NIR.mp4")
cap = cv2.VideoCapture(video)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {w}x{h}, {n} frames")

frames = []
while True:
    ret, f = cap.read()
    if not ret:
        break
    frames.append(f[:, :, 0] if f.ndim == 3 else f)
cap.release()

# For each consecutive pair, compute residual and find peaks
print(f"\nPer-frame residual analysis (homography):")
print(f"{'frame':>5}  {'res_mean':>8}  {'res_max':>7}  {'bg_mean':>7}  {'bg_std':>6}  "
      f"{'peaks>5':>7}  {'peaks>8':>7}  {'peaks>12':>8}  {'top5_vals':>30}  {'top5_locs':>50}")

peak_map = np.zeros((h, w), dtype=np.float32)
all_peaks = []

for i in range(1, len(frames)):
    residual, H, ni = motion_compensated_residual(frames[i-1], frames[i], method="homography")
    rm = residual.mean()
    rmax = residual.max()
    bm = frames[i].mean()
    bs = frames[i].std()

    # Find peaks: pixels above a low absolute threshold
    hot = np.where(residual > 5)
    n5 = len(hot[0])
    hot8 = np.where(residual > 8)
    n8 = len(hot8[0])
    hot12 = np.where(residual > 12)
    n12 = len(hot12[0])

    # Top 5 brightest residual pixels
    flat = residual.ravel()
    top_idx = np.argpartition(flat, -5)[-5:]
    top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]
    top_vals = flat[top_idx]
    top_ys = top_idx // w
    top_xs = top_idx % w
    top_locs = [(int(x), int(y)) for x, y in zip(top_xs, top_ys)]

    peak_map = np.maximum(peak_map, residual.astype(np.float32))

    if i % 10 == 0 or rmax > 15 or n8 > 10:
        print(f"{i:5d}  {rm:8.2f}  {rmax:7d}  {bm:7.1f}  {bs:6.1f}  "
              f"{n5:7d}  {n8:7d}  {n12:8d}  {str(top_vals.tolist()):>30}  {str(top_locs):>50}")

    for x, y, v in zip(top_xs, top_ys, top_vals):
        if v > 8:
            all_peaks.append((i, int(x), int(y), float(v), float(bm)))

# Cluster all peaks spatially to find distinct objects
print(f"\n--- All bright peaks (residual > 8) across video: {len(all_peaks)} ---")

# Group by rough trajectory: bin by (y, x_velocity_adjusted)
from sklearn.cluster import DBSCAN
if all_peaks:
    # Use (frame, x, y) but stretch frame axis to match spatial velocity
    arr = np.array([(p[0], p[1], p[2]) for p in all_peaks], dtype=np.float64)
    # Expected vx ~ 5.4 px/frame: transform x → x - 5.4*frame so co-moving targets cluster
    vx_est = 5.4
    vy_est = 1.3
    co_moving = np.column_stack([
        arr[:, 1] - vx_est * arr[:, 0],  # x adjusted
        arr[:, 2] - vy_est * arr[:, 0],  # y adjusted
    ])
    clustering = DBSCAN(eps=15, min_samples=3).fit(co_moving)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}
    print(f"Found {len(unique_labels)} trajectory clusters (DBSCAN eps=15, min_samples=3)")
    for cid in sorted(unique_labels):
        members = [all_peaks[j] for j in range(len(all_peaks)) if labels[j] == cid]
        frames_in = [m[0] for m in members]
        xs = [m[1] for m in members]
        ys = [m[2] for m in members]
        vals = [m[3] for m in members]
        bgs = [m[4] for m in members]
        print(f"  Cluster {cid}: {len(members)} peaks, frames {min(frames_in)}-{max(frames_in)}, "
              f"x={min(xs)}-{max(xs)}, y={min(ys)}-{max(ys)}, "
              f"peak_vals={min(vals):.0f}-{max(vals):.0f}, bg_mean={np.mean(bgs):.0f}")

    noise = [all_peaks[j] for j in range(len(all_peaks)) if labels[j] == -1]
    print(f"  Noise points: {len(noise)}")

# Save the max-projected peak map
peak_vis = np.clip(peak_map * 10, 0, 255).astype(np.uint8)
cv2.imwrite(str(Path(__file__).resolve().parent / "peak_map.png"), peak_vis)
print(f"\nPeak map saved to _nadir_output/peak_map.png")
