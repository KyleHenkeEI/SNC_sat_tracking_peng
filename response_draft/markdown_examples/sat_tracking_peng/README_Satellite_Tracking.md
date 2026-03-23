# 🛰️ Satellite Tracking Pipeline

A Python-based computer vision pipeline for detecting and tracking satellites (or other moving point sources) in video footage. The system uses background subtraction, clustering-based detection, Kalman filtering for smooth trajectory estimation, and Hungarian algorithm for multi-object tracking.

## Terminal-only usage

**Multi-tracker comparison** (all `run_*.py` scripts, `run_tracking_comparison.py`, output layout, and flags) is documented in **[README_Tracker_Comparison.md](README_Tracker_Comparison.md)**.

Tracker code lives in the `tracking_core/` package. Run a single variant:

```bash
python run_advanced.py --input-video path/to/input.mp4 --output-dir path/to/tracker_out
```

Compare multiple trackers (subprocesses + aggregate CSV/JSON/plot + comparison videos):

```bash
python run_tracking_comparison.py --input-video path/to/input.mp4 --trackers "Advanced baseline" "Current PMB" --output-dir path/to/results
```

Each run creates `results/<run_name>/<output_tag>/output_video.mp4`, `tracks.txt`, and `summary.json`. The comparison runner also writes `comparison_summary.csv`, `comparison_summary.json`, `comparison_manifest.json`, `comparison_plot.png`, `comparison_grid.mp4`, and `comparison_with_original.mp4` (when at least two trackers succeed).

Install dependencies: `pip install -r requirements.txt` or `.\install_requirements.ps1` on Windows.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Step 1: Background Computation](#step-1-background-computation)
3. [Step 2: Background Subtraction & Thresholding](#step-2-background-subtraction--thresholding)
4. [Step 3: Point Source Detection with DBSCAN](#step-3-point-source-detection-with-dbscan)
5. [Step 4: Track Association (Hungarian Algorithm)](#step-4-track-association-hungarian-algorithm)
6. [Step 5: Kalman Filter for Position & Velocity Estimation](#step-5-kalman-filter-for-position--velocity-estimation)
7. [Step 6: Confidence Scoring](#step-6-confidence-scoring)
8. [Step 7: Motion Filtering](#step-7-motion-filtering)
9. [Step 8: Track Lifecycle Management](#step-8-track-lifecycle-management)
10. [Output Files](#output-files)
11. [Parameter Reference](#parameter-reference)

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SATELLITE TRACKING PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   VIDEO INPUT                                                               │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────┐                                                   │
│   │  1. BACKGROUND      │  Compute median/mode of all frames                │
│   │     COMPUTATION     │  → Creates static background image                │
│   └─────────────────────┘                                                   │
│       │                                                                     │
│       ▼ (for each frame)                                                    │
│   ┌─────────────────────┐                                                   │
│   │  2. BACKGROUND      │  |frame - background| → difference image         │
│   │     SUBTRACTION     │  threshold → binary mask                         │
│   └─────────────────────┘                                                   │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────┐                                                   │
│   │  3. DBSCAN          │  Cluster white pixels → point detections         │
│   │     CLUSTERING      │  Output: centroids, bounding boxes, areas        │
│   └─────────────────────┘                                                   │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────┐                                                   │
│   │  4. HUNGARIAN       │  Match detections to existing tracks             │
│   │     ASSOCIATION     │  Minimize total assignment cost                  │
│   └─────────────────────┘                                                   │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────┐                                                   │
│   │  5. KALMAN FILTER   │  Predict → Update cycle                          │
│   │     UPDATE          │  Smooth position, estimate velocity              │
│   └─────────────────────┘                                                   │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────┐                                                   │
│   │  6. CONFIDENCE      │  Score based on consecutive detections           │
│   │     SCORING         │  Penalize missed frames                          │
│   └─────────────────────┘                                                   │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────┐                                                   │
│   │  7. MOTION FILTER   │  Remove stationary objects (stars, hot pixels)   │
│   │                     │  Keep only tracks with speed > min_speed         │
│   └─────────────────────┘                                                   │
│       │                                                                     │
│       ▼                                                                     │
│   VIDEO OUTPUT + TRACK LOG (.txt)                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Background Computation

### Purpose
Create a reference image representing the static background (stars, sensor noise patterns) that remains constant across frames. Moving objects (satellites) will deviate from this background.

### Method
The pipeline supports two methods:

| Method | Description | Best For |
|--------|-------------|----------|
| `median` | Pixel-wise median across all frames | Most cases; robust to outliers |
| `statistical` | Mode (most frequent value) per pixel | Very short videos (<100 frames) |

### How It Works
```python
# Stack all frames into a 3D array: (num_frames, height, width)
stacked_frames = np.stack(frames, axis=0)

# Compute median along the time axis
background = np.median(stacked_frames, axis=0)
```

### Why Median?
- **Robust to moving objects**: A satellite passing through a pixel only affects a few frames, so the median ignores it
- **Handles sensor noise**: Random noise averages out
- **Simple and effective**: No tuning required

---

## Step 2: Background Subtraction & Thresholding

### Purpose
Isolate pixels that differ significantly from the background — these are candidate moving objects.

### Process

```
Frame ──────┐
            │
            ▼
      ┌───────────┐
      │  absdiff  │  Compute absolute difference: |frame - background|
      └───────────┘
            │
            ▼
      ┌───────────┐
      │ threshold │  If diff > bg_threshold → 255 (white), else → 0 (black)
      └───────────┘
            │
            ▼
      Binary Mask (white = foreground)
```

### Code
```python
diff = cv2.absdiff(frame, self.background)
_, binary = cv2.threshold(diff, self.bg_threshold, 255, cv2.THRESH_BINARY)
```

### Parameter: `bg_threshold`
| Value | Effect |
|-------|--------|
| 5-10 | Very sensitive, detects faint objects, but more noise |
| 15-20 | Balanced sensitivity (recommended) |
| 25-30 | Only bright objects, cleaner but may miss faint satellites |

---

## Step 3: Point Source Detection with DBSCAN

### Purpose
Group connected/nearby white pixels in the binary mask into distinct detections. Each cluster represents one potential object.

### Why DBSCAN?
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is ideal for this task because:
- **No predefined cluster count**: We don't know how many satellites are in a frame
- **Handles noise**: Isolated pixels (label = -1) are automatically filtered out
- **Arbitrary shapes**: Works for point sources that may span a few pixels

### Algorithm
```
1. Find all white pixels in binary mask
2. Run DBSCAN clustering on pixel coordinates
3. For each cluster:
   - Compute centroid (mean x, mean y)
   - Compute bounding box (min/max coordinates)
   - Compute area (pixel count)
```

### Parameters

| Parameter | Description | Effect |
|-----------|-------------|--------|
| `cluster_eps` | Maximum distance between pixels to be in the same cluster | Higher = merge nearby pixels; Lower = split into separate detections |
| `cluster_min_samples` | Minimum pixels required to form a cluster | 1 = single pixels count; Higher = filters isolated noise |

### Output per Detection
```python
{
    'centroid': (x, y),        # Center of mass
    'bbox': (x, y, w, h),      # Bounding box
    'area': N                   # Number of pixels in cluster
}
```

---

## Step 4: Track Association (Hungarian Algorithm)

### Purpose
Match new detections to existing tracks across frames. This is the core of multi-object tracking.

### Challenge
Given N existing tracks and M new detections, find the optimal assignment that minimizes total "cost" (distance).

### Solution: Hungarian Algorithm
The Hungarian algorithm (via `scipy.optimize.linear_sum_assignment`) finds the globally optimal assignment in O(n³) time.

### Cost Matrix Construction

```
                    Detections
                 D₁    D₂    D₃    D₄
              ┌─────┬─────┬─────┬─────┐
         T₁   │ 12  │ 45  │ 1e9 │ 8   │
Tracks   T₂   │ 1e9 │ 15  │ 22  │ 1e9 │
         T₃   │ 30  │ 1e9 │ 1e9 │ 5   │
              └─────┴─────┴─────┴─────┘

Cost = distance between predicted position and detection
1e9 = "infinite" cost (gated out, too far)
```

### Velocity-Gated Association
To prevent incorrect associations, the algorithm uses **gating**:

1. **Kalman Prediction**: Where we expect the track to be
2. **Linear Extrapolation**: Last position + velocity
3. **Gate Distance**: Only consider detections within a dynamic radius

```python
gate_distance = min(max_distance, max(15, speed * velocity_gate_factor + 10))
```

Fast-moving objects have tighter gates near their predicted position; slow objects have looser gates.

### Motion Consistency Check
Before accepting an association, verify the implied acceleration is realistic:

```python
accel = |new_velocity - current_velocity|
if accel > max_acceleration:
    reject association  # Would require unrealistic motion
```

---

## Step 5: Kalman Filter for Position & Velocity Estimation

### Purpose
1. **Smooth noisy detections**: Raw pixel detections jitter; Kalman filtering produces smooth trajectories
2. **Estimate velocity**: Even though we only measure position, the filter infers velocity
3. **Predict during occlusions**: When a detection is missed, predict where the object should be

### State Vector
```
x = [x_position, y_position, x_velocity, y_velocity]ᵀ
```

### System Model (Constant Velocity)
```
State Transition Matrix F:
┌                    ┐
│ 1  0  dt  0  │   x' = x + vx*dt
│ 0  1  0   dt │   y' = y + vy*dt
│ 0  0  1   0  │   vx' = vx
│ 0  0  0   1  │   vy' = vy
└                    ┘

Measurement Matrix H:
┌            ┐
│ 1  0  0  0 │   We only observe position (x, y)
│ 0  1  0  0 │
└            ┘
```

### Predict-Update Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                      KALMAN FILTER CYCLE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   PREDICT (every frame)                                     │
│   ─────────────────────                                     │
│   x̂ₖ|ₖ₋₁ = F·x̂ₖ₋₁        (predict state)                   │
│   Pₖ|ₖ₋₁ = F·Pₖ₋₁·Fᵀ + Q  (predict covariance)             │
│                                                             │
│   UPDATE (when detection matched)                           │
│   ─────────────────────────────────                         │
│   y = z - H·x̂ₖ|ₖ₋₁        (innovation/residual)            │
│   S = H·Pₖ|ₖ₋₁·Hᵀ + R     (innovation covariance)          │
│   K = Pₖ|ₖ₋₁·Hᵀ·S⁻¹       (Kalman gain)                    │
│   x̂ₖ = x̂ₖ|ₖ₋₁ + K·y       (updated state)                  │
│   Pₖ = (I - K·H)·Pₖ|ₖ₋₁   (updated covariance)             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Parameters

| Parameter | Description | Effect |
|-----------|-------------|--------|
| `process_noise` (Q) | How much we expect velocity to change | Lower = smoother, slower to adapt; Higher = more responsive |
| `measurement_noise` (R) | How much we trust raw detections | Lower = follow detections closely; Higher = trust predictions more |

### Speed Estimation
Velocity is part of the state vector and updated every cycle:
```python
speed = sqrt(vx² + vy²)  # pixels per frame
```

---

## Step 6: Confidence Scoring

### Purpose
Quantify how confident we are that a track is a real satellite vs. noise. Confidence is visualized as color:

```
🔴 Red (0.0-0.3)     → New or uncertain track
🟡 Yellow (0.3-0.7)  → Building confidence
🟢 Green (0.7-1.0)   → High confidence, established track
```

### Scoring Formula

```python
if consecutive_detections < min_confidence_frames:
    # Still building initial confidence
    base_confidence = 0.15 + (consecutive / min_confidence_frames) * 0.25
else:
    # Ramp up to full confidence
    effective = min(consecutive, max_confidence_frames)
    base_confidence = 0.4 + 0.6 * (effective - min_frames) / (max_frames - min_frames)

# Penalize frames without detection
age_penalty = min(0.4, (age - 1) * 0.08)

final_confidence = base_confidence - age_penalty
```

### Behavior
| Scenario | Confidence |
|----------|------------|
| First detection | 0.15 (red) |
| 5 consecutive detections | ~0.4 (orange) |
| 25+ consecutive detections | 1.0 (green) |
| 1 missed frame | -0.08 penalty |
| 5 missed frames | -0.40 penalty |

---

## Step 7: Motion Filtering

### Purpose
Satellites move; stars and hot pixels don't. This step filters out stationary detections.

### Criteria
A track is considered "moving" if **either**:
1. **Instantaneous speed** ≥ `min_speed` (from Kalman filter)
2. **Total displacement** from start > `min_speed × total_frames × 0.5`

### Why Both Checks?
- **Speed check**: Catches fast-moving objects immediately
- **Displacement check**: Catches slow-moving objects that have traveled far over time

### Parameter: `min_speed`
| Value | Effect |
|-------|--------|
| 0 | Show all tracks (no filtering) |
| 0.5 | Filter very slow/stationary objects |
| 2+ | Only show clearly moving objects |

---

## Step 8: Track Lifecycle Management

### Track States

```
┌─────────────┐     detection      ┌─────────────┐
│   NEW       │ ────────────────▶  │   ACTIVE    │
│  (created)  │                    │  (tracking) │
└─────────────┘                    └─────────────┘
                                         │
                         no detection    │
                         for N frames    │
                                         ▼
                                   ┌─────────────┐
                                   │    LOST     │
                                   │ (predict    │
                                   │  only)      │
                                   └─────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
             detection found                          timeout exceeded
             (re-identification)                              │
                    │                                         ▼
                    ▼                                   ┌─────────────┐
              ┌─────────────┐                          │   DELETED   │
              │   ACTIVE    │                          └─────────────┘
              │  (resumed)  │
              └─────────────┘
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `track_timeout` | Frames without detection before moving to "lost" state |
| `lost_track_timeout` | Frames to keep lost tracks before deletion |

### Re-Identification
Lost tracks are not immediately deleted. The system:
1. Predicts where the lost track would be (using last known velocity)
2. If a new detection appears near the prediction, the track is resumed
3. This handles temporary occlusions or missed detections

---

## Output Files

### 1. Video Output (`*_output.mp4`)
Side-by-side comparison:
- **Left**: Original frame
- **Right**: Annotated frame with:
  - Bounding boxes (color = confidence)
  - Trajectory tails
  - Current position markers

### 2. Track Log (`*_tracks.txt`)
```
# Satellite Tracking Log
# Input: path/to/video.mp4
# Format: frame, track_id, x, y, confidence, speed, vx, vy
#======================================================================
250, 1, 523, 412, 0.4500, 2.3456, 1.2345, 2.0123
251, 1, 525, 414, 0.5100, 2.4012, 1.3456, 2.1234
252, 1, 528, 416, 0.5800, 2.3890, 1.4000, 2.0500
...
```

| Column | Description |
|--------|-------------|
| `frame` | Frame number |
| `track_id` | Unique identifier for this track |
| `x`, `y` | Position (Kalman-smoothed if enabled) |
| `confidence` | Track confidence (0-1) |
| `speed` | Speed in pixels/frame |
| `vx`, `vy` | Velocity components (pixels/frame) |

---

## Parameter Reference

### Detection Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `bg_threshold` | 15 | Pixel difference to count as foreground |
| `bg_method` | 'median' | Background computation method |
| `cluster_eps` | 3 | DBSCAN max distance for clustering |
| `cluster_min_samples` | 1 | DBSCAN min pixels per cluster |

### Tracking Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_distance` | 50 | Max pixels a track can move per frame |
| `process_noise` | 5.0 | Kalman filter process noise (Q) |
| `measurement_noise` | 10.0 | Kalman filter measurement noise (R) |
| `track_timeout` | 15 | Frames before track becomes "lost" |
| `lost_track_timeout` | 30 | Frames before lost track is deleted |

### Display & Filtering Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_track_length` | 3 | Min detections to display a track |
| `min_speed` | 0.5 | Min speed to display (filters stars) |
| `max_acceleration` | 20.0 | Max velocity change per frame |
| `use_kalman_display` | True | Use smoothed positions for display |
| `track_history_length` | 50 | Trajectory tail length |

---

## Dependencies

```
opencv-python
numpy
scikit-learn
scipy
```

---

## Quick Start

```python
from satellite_tracker import SatelliteTrackingPipeline

pipeline = SatelliteTrackingPipeline(
    input_video_path="input.mp4",
    output_video_path="output.mp4",
    bg_threshold=15,
    min_speed=2,
    save_track_log=True
)

results = pipeline.run()
```

---

## License

MIT License

