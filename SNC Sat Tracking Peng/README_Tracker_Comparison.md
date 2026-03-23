# Multi-tracker comparison (terminal)

This document describes how to **run and compare several tracking algorithms** on the same video from the command line.  

For a **deep dive into the adaptive Kalman / “Advanced” pipeline only** (background subtraction, association, scoring steps), see **[README_Satellite_Tracking.md](README_Satellite_Tracking.md)**.

---

## What you get

| Piece | Role |
|--------|------|
| **`tracking_core/`** | Shared library: `AdvancedSatelliteTracker`, `PoissonMultiBernoulliTracker`, variant classes, `TRACKER_VARIANTS` registry. |
| **`run_*.py`** (repo root) | One script per tracker; same CLI: `--input-video`, `--output-dir`, optional `--quiet`. |
| **`run_tracking_comparison.py`** | Runs selected (or all) tracker scripts, aggregates metrics, builds **comparison plots and videos**. |

---

## What is actually implemented (exact code map)

The comparison does **not** ship many independent trackers from separate papers. It ships **two full pipelines** in **`tracking_core/advanced.py`** and **`tracking_core/pmb.py`**, plus **variant classes** in **`tracking_core/variants.py`** (and **`TrueTbdPmTracker`** in **`pmb.py`**) that **subclass** one of the two pipelines and override specific methods. Advanced-side variant logic lives in **`variants.py`** unless noted otherwise.

### Base pipelines (where the bulk of the code is)

| Module | Class | What it does (detect → track) |
|--------|--------|-------------------------------|
| **`advanced.py`** | `AdvancedSatelliteTracker` | Median background, absdiff + threshold, optional morphology, **DBSCAN** on foreground pixels → centroids; **Hungarian** (`linear_sum_assignment`) on a Mahalanobis / distance cost matrix; per-track **`AdaptiveKalmanFilter2D`**; lost-track re-ID; confidence and motion gates. |
| **`pmb.py`** | `PoissonMultiBernoulliTracker` | Same style **preprocessing** → binary mask; clustered **detections**; **Bernoulli** components with Kalman-style predict/update; **likelihood** matrix + existence updates; prune/merge; side-by-side render. |
| **`pmb.py`** | `TrueTbdPmTracker` | Subclasses **`PoissonMultiBernoulliTracker`**: **soft fused residual map** (temporal max/mean, **no** global binary mask in the loop); birth proposals from **local maxima** with a **per-frame percentile** floor; association likelihoods scaled by **local integrated evidence**. Intended as a **true TBD-style** passive-optical baseline vs threshold-then-detect. |

### Registry entry → class → what was added on top

| Name in CLI / `--trackers` | Python class | Inherits from | What this entry **actually** changes |
|----------------------------|--------------|---------------|--------------------------------------|
| **Advanced baseline** | `AdvancedSatelliteTracker` | — | Full **`advanced.py`** pipeline only. |
| **Current PMB** | `PoissonMultiBernoulliTracker` | — | Full **`pmb.py`** pipeline only. |
| **PMB large/fast** | `PmbLargeFastTracker` | PMB | Subclasses **`AdaptiveRFSFamilyTracker`**: **OR** of standard and softer absdiff masks, **percentile-based supplemental** blob proposals (with relaxed intensity for large regions), **stronger births** for bright/large detections, **gentler pruning / miss decay** for fast or large established components, and short **display hysteresis** + **smoothed centroids** to cut flicker. Registry still raises **`max_detection_area`** and relaxes accel / confirmation vs base adaptive RFS. **Does not** change **`Current PMB`**. |
| **IMM adaptive** | `IMMAdaptiveMotionTracker` | Advanced | **Heuristic “modes”** (`search` / `cruise` / `maneuver` / `fast`) from speed + Kalman innovation; each mode scales **Mahalanobis gate**, **accel**, **turn**, **speed** limits. **`compute_association_costs`** and **`check_motion_validity`** use those profiles. **Not** a full IMM with multiple explicit motion models and mixing probabilities. |
| **JPDA lite** | `JPDALiteTracker` | Advanced | Custom **`associate_tracks`**: greedy row-wise matching with optional **soft blend** of multiple nearby detections (default **`jpda_max_soft_neighbors=1`** = **no** centroid blend; uses **Mahalanobis + pixel** gates like Advanced). **Not** full JPDA. |
| **MHT lite** | `MHTLiteTracker` | Advanced | Custom **`associate_tracks`**: **Hungarian** with post-gates; for **unmatched** tracks, if two candidates are **within `ambiguity_margin`**, uses a **mostly-primary** centroid blend (**~92/8**) and marks **`hypothesis_state` = tentative**. **Not** a real MHT tree or N-scan pruning. |
| **Particle assisted** | `ParticleAssistedTracker` | Advanced | Per-track **bootstrap particle cloud** (position/velocity); **resampling** vs measurement; **`compute_association_costs`** blends **Kalman prediction** with **particle mean**; **`associate_tracks`** defers to base then **updates particles**. Still the same detection + base association structure as Advanced. |
| **Track-before-detect** | `TrueTbdPmTracker` | PMB | **Soft-evidence TBD:** fused **absdiff** maps (same temporal fusion weights as the old baseline) **without** converting the full frame to a single binary mask; **local peak** proposals; PMB **likelihood** includes a **local evidence** scale factor. Compare against **Temporal-accumulation TBD (approx)** for thresholded behavior. |
| **Temporal-accumulation TBD (approx)** | `TrackBeforeDetectTracker` | Advanced | Legacy baseline: only **`preprocess_frame`** overridden — buffers last **`tbd_window`** **absdiff** maps, fuses with **`0.6 * max + 0.4 * mean`**, then **hard thresholds** (with **`tbd_gain`**) and runs the usual **detect-then-track** pipeline. **Not** classical TBD on raw maps without thresholding. |
| **Adaptive RFS family** | `AdaptiveRFSFamilyTracker` | PMB | **`detect_objects`**: adds **bbox** and **area** on detections (vs simpler PMB listing). **`update_components`**: **scales** accel / turn / speed gates by **`_association_scale`** (size + speed). **`prune_and_merge`** / **`get_confirmed_tracks`**: **dynamic** min track length and display confidence for fast/large components. Still the same **single-layer Bernoulli PMB** update as base PMB, **not** GLMB / LMB / full PMBM. |

### Qualitative scores in CSV/JSON

The columns **Noise / Motion / Clutter / Compute** (1–5) come from the static **`scores`** dict on each entry in **`TRACKER_VARIANTS`** inside **`variants.py`**. They are **labels for plotting**, not measured from your video. **Measured** fields are things like **`Runtime (s)`**, frame count, detection count, and track count from each run’s **`summary.json`**.

### Where to read the real algorithms

- **Advanced family (baseline + six variants):** `tracking_core/advanced.py` + `tracking_core/variants.py` (classes above).
- **PMB + adaptive RFS + true TBD:** `tracking_core/pmb.py` (`PoissonMultiBernoulliTracker`, `TrueTbdPmTracker`) + `AdaptiveRFSFamilyTracker` in `tracking_core/variants.py`.
- **Registry and default hyperparameters per entry:** `TRACKER_VARIANTS`, `ADVANCED_BASELINE_KWARGS`, `PMB_BASELINE_KWARGS` at the bottom of **`variants.py`**.

---

## Prerequisites

From the project folder:

```powershell
cd "D:\path\to\SNC Sat Tracking Peng"
python -m pip install -r requirements.txt
```

Use **`python`** (the same interpreter you used for `pip`). On some systems the `py` launcher is not installed; that’s normal.

---

## List available trackers

```powershell
python run_tracking_comparison.py --list-trackers
```

Exact names (in quotes when they contain spaces) are used with `--trackers`.

---

## Run a single tracker

Each launcher writes **`output_video.mp4`**, **`tracks.txt`**, and **`summary.json`** under `--output-dir`.

```powershell
python run_advanced.py --input-video "D:\data\clip.mp4" --output-dir "D:\out\run1\advanced_baseline"
```

Other entry points:

| Script | Tracker name |
|--------|----------------|
| `run_advanced.py` | Advanced baseline |
| `run_pmb.py` | Current PMB |
| `run_pmb_large_fast.py` | PMB large/fast |
| `run_imm.py` | IMM adaptive |
| `run_jpda.py` | JPDA lite |
| `run_mht.py` | MHT lite |
| `run_particle.py` | Particle assisted |
| `run_tbd.py` | Track-before-detect (true soft-evidence TBD, PMB) |
| `run_tbd_approx.py` | Temporal-accumulation TBD (approx) |
| `run_adaptive_rfs.py` | Adaptive RFS family |

Add **`--quiet`** to reduce console noise from the tracker itself.

---

## Run a comparison (orchestrator)

### All trackers on one video

```powershell
python run_tracking_comparison.py `
  --input-video "D:\data\clip.mp4" `
  --output-dir "D:\out\comparison_results"
```

If you omit **`--trackers`**, **all** registered trackers run **one after another** (can take a long time on long clips).

### All trackers, first `.mp4` in a folder

Videos must sit **directly** in the folder (not only in subfolders). The first file when names are sorted **case-insensitively** is chosen.

```powershell
python run_tracking_comparison.py `
  --input-dir "D:\data\my_folder_of_mp4s" `
  --output-dir "D:\out\comparison_results" `
  --run-name my_experiment
```

### Subset of trackers

```powershell
python run_tracking_comparison.py `
  --input-video "D:\data\clip.mp4" `
  --trackers "Advanced baseline" "Current PMB" "IMM adaptive" `
  --output-dir "D:\out\comparison_results"
```

### Named run folder

By default a timestamped subfolder is created under `--output-dir`. Fix the name with **`--run-name`**:

```powershell
--run-name parallel_run5_tile0
```

---

## Command-line options (`run_tracking_comparison.py`)

| Option | Meaning |
|--------|---------|
| `--input-video PATH` | Explicit input clip (wins over `--input-dir` if both set). |
| `--input-dir PATH` | Folder; uses first `*.mp4` (sorted by name, case-insensitive). |
| `--output-dir PATH` | Root for outputs; default: `comparison_results` under the repo. |
| `--run-name NAME` | Subfolder under `--output-dir`; default: `run_YYYYMMDD_HHMMSS`. |
| `--trackers NAME ...` | Which trackers to run; default: all. |
| `--list-trackers` | Print names and exit (no video needed). |
| `--skip-run` | Write registry-based summaries only; no tracker subprocesses. |
| `--no-comparison-videos` | Skip `comparison_grid.mp4` and `comparison_with_original.mp4`. |
| `--capture-output` | Buffer child stdout/stderr and print after each tracker (no live stream). **Default** is live streaming to your terminal. |

Child processes run with **`PYTHONUNBUFFERED=1`** so frame progress appears sooner.

---

## Output layout

After a comparison, you get a run directory:

```text
<output-dir>/<run-name>/
├── comparison_summary.csv          # Spreadsheet-friendly
├── comparison_summary.json         # Same rows as JSON
├── comparison_manifest.json        # Paths + metadata
├── comparison_plot.png             # Heatmap / scatter / runtime (needs matplotlib)
├── comparison_grid.mp4             # Horizontal strip of tracker outputs only (≥2 successes)
├── comparison_with_original.mp4    # Raw input on top + tracker overlays in a grid below (≥2 successes)
├── advanced_baseline/
│   ├── output_video.mp4
│   ├── tracks.txt
│   └── summary.json
├── pmb_baseline/
│   └── ...
└── ...                             # One folder per tracker output_tag
```

**`summary.json`** (per tracker) includes status, runtime, frame/detection counts, and paths to video and track log.

---

## Qualitative scores vs measured runtime

The CSV/JSON include both **fixed qualitative scores** (noise / motion / clutter / compute, 1–5) from the registry in `tracking_core/variants.py` and **measured** fields when a run succeeds (`Runtime (s)`, frames, detections, tracks). The plot uses those together.

---

## Troubleshooting

| Issue | What to do |
|--------|------------|
| `ModuleNotFoundError: No module named 'cv2'` | `python -m pip install -r requirements.txt` using the **same** `python` you use to run scripts. |
| `py` is not recognized | Use `python` instead of `py`. |
| No video picked from `--input-dir` | Ensure at least one `.mp4` is **in that folder** (not only in subfolders), or use `--input-video`. |
| Comparison videos missing | Need **at least two** trackers with status **ok** and valid `output_video.mp4`. Use `--no-comparison-videos` only if you want to skip them. |
| No live progress | Remove `--capture-output` if you added it; run in Cursor/VS Code terminal or PowerShell. |

---

## Where to change algorithms

- **Registry and baseline kwargs:** `tracking_core/variants.py` (`TRACKER_VARIANTS`, `ADVANCED_BASELINE_KWARGS`, `PMB_BASELINE_KWARGS`).
- **Advanced family implementation:** `tracking_core/advanced.py`.
- **PMB / Bernoulli:** `tracking_core/pmb.py`.
- **Comparison montage codec/layout:** `tracking_core/comparison_video.py`.

After changing the registry, keep **`TRACKER_SCRIPT_BY_NAME`** in `run_tracking_comparison.py` in sync if you add new named scripts.

---

## Related files

- `requirements.txt` / `install_requirements.ps1` — dependencies.
- `README_Satellite_Tracking.md` — Advanced tracker pipeline narrative (single-algorithm focus).
