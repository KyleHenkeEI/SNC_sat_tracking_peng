# Satellite Video Tracking — Algorithm Overview (RFI)

This document describes the **algorithms and methods** implemented in the *SNC Sat Tracking Peng* codebase for passive-optical satellite (or point-source) tracking in video. It is written for **requests for information (RFI)** and external technical review.

**Math in this file:** Display and inline formulas use `$...$` and `$$...$$` so they render in **Cursor / VS Code** Markdown preview (enable **Markdown › Math** if needed). Plain **GitHub.com** does not render LaTeX in `.md` files; export to PDF/HTML with a LaTeX-capable tool if you need math there.

---

## 1. Executive summary

The project provides **ten named tracker configurations** registered in `tracking_core/variants.py` (`TRACKER_VARIANTS`). Under the hood there are **two primary pipelines**:


| Family                           | Core module                                            | Idea                                                                                                                                                                                                                                                                                                    |
| -------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Advanced (detect-then-track)** | `tracking_core/advanced.py`                            | Background subtraction → thresholded foreground → **DBSCAN** point detections → **Hungarian** assignment to tracks → **adaptive 2D Kalman** filtering, confidence, and motion gates.                                                                                                                    |
| **PMB family**                   | `tracking_core/pmb.py` (+ subclasses in `variants.py`) | Same style **preprocessing** (for most variants) → detections → **multi-target Bernoulli-style** filtering: each hypothesized target is a **Bernoulli component** with existence probability $r$, Kalman state, Gaussian **likelihood**, survival/miss updates, births, pruning, and display filtering. |


Several entries are **variants** that **subclass** one of these pipelines and override specific steps (association, preprocessing, detection, or PMB update scaling). Names such as **“JPDA lite”**, **“MHT lite”**, or **“Adaptive RFS family”** reflect **design inspiration**; they are **not** full textbook implementations of those methods unless stated below.

**Outputs** (per run): annotated video, `tracks.txt`, and `summary.json` (see `tracking_core/tracker_cli.py` and `README_Tracker_Comparison.md`).

---

## 2. Terminology (for RFI readers)

- **Implemented algorithm** — Code path executed at runtime (e.g. `PoissonMultiBernoulliTracker.run`).
- **Variant** — A subclass or configuration that changes part of a base pipeline while reusing the rest.
- **Literature-inspired / “lite”** — Naming aligns with classical methods (JPDA, MHT, IMM, GLMB/LMB) but the implementation may use **heuristics** or **greedy** steps instead of the full formal filter.

---

## 3. Poisson multi-Bernoulli (PMB) family — detailed description

The canonical implementation is `**PoissonMultiBernoulliTracker`** in `tracking_core/pmb.py`, exposed in the CLI as **“Current PMB”**. Related trackers (**Adaptive RFS family**, **PMB large/fast**, **Track-before-detect**) extend or specialize this path.

### 3.1 Conceptual model (theory-forward)

In random finite set (RFS) tracking, a **multi-Bernoulli** representation describes the multi-target posterior as a set of **independent Bernoulli** targets: each target $i$ has **existence probability** $r_i \in [0,1]$ and a **spatial state density** (here approximated by a **Gaussian** via a Kalman filter on $\mathbf{x} \in \mathbb{R}^4$).

A **Poisson multi-Bernoulli mixture (PMBM)** / related filters combine **unknown number of births** (often Poisson) with **Bernoulli** tracks and perform **data association** across hypotheses. **Standard PMB/PMBM** updates require structured handling of **missed detections**, **clutter**, and **multi-target association** (often via labeled RFS or explicit hypothesis trees).

### 3.2 What this codebase implements

The implementation maintains a list of `**BernoulliComponent`** objects. Each component stores:

- **Existence probability** $r$ (attribute `r`).
- **State** $\mathbf{x} = [x, y, v_x, v_y]^\top$ (constant-velocity model, discrete time step $dt = 1$ per frame).
- **Covariance** $\mathbf{P}$, with fixed **process noise** $\mathbf{Q}$ and **measurement noise** $\mathbf{R} = 2\mathbf{I}_2$ inside `BernoulliComponent` (not exposed as tracker hyperparameters).

**Detection front-end (Current PMB):**

1. **Background:** median of subsampled frames (`compute_background`).
2. **Preprocessing:** absolute difference vs background, Gaussian blur, binary threshold (`preprocess_frame`).
3. **Measurements:** connected bright pixels clustered by **DBSCAN**; centroids and peak intensities are detections (`detect_objects`).

**Time update (predict):** for each component,

$$
\mathbf{x}*{k|k-1} = \mathbf{F}\mathbf{x}*{k-1}, \qquad
\mathbf{P}*{k|k-1} = \mathbf{F}\mathbf{P}*{k-1}\mathbf{F}^\top + \mathbf{Q},
$$

$$
r_{k|k-1} = p_s r_{k-1},
$$

where $p_s$ is `**survival_prob`** in code.

**Measurement update:** let $\mathbf{z}*j \in \mathbb{R}^2$ be detection $j$. The **predicted measurement** is $\hat{\mathbf{z}} = \mathbf{H}\mathbf{x}*{k|k-1}$, innovation $\boldsymbol{\nu}_j = \mathbf{z}*j - \hat{\mathbf{z}}$, innovation covariance $\mathbf{S} = \mathbf{H}\mathbf{P}*{k|k-1}\mathbf{H}^\top + \mathbf{R}$. The **Gaussian likelihood** is

$$
L_{ij} = \mathcal{N}\left(\mathbf{z}_j; \hat{\mathbf{z}}, \mathbf{S}\right).
$$

Associations that violate **physical gates** (speed, acceleration, turn rate) are zeroed (`check_physical_constraints`).

For a given component $i$ and candidate detection $j$, the code updates existence with a **Bayes-rule-style ratio** (detection vs clutter) of the form

$$
r_{ij}^{+} = \frac{p_d r L_{ij}}{p_d r L_{ij} + \lambda_c (1 - r) + \varepsilon},
$$

where $p_d$ is `**detection_prob`**, $\lambda_c$ is a **clutter intensity** derived from `**clutter_rate`** (normalized by a fixed $128 \times 128$ factor in code), and $\varepsilon$ is a small numerical floor.

**Data association** is **greedy per component**: components are processed in list order; each chooses the unused detection that yields the largest $r_{ij}^{+}$ above a minimum threshold; otherwise a **miss** update $r \leftarrow (1-p_d) r$ is applied. **Unused detections** spawn **new Bernoulli births** with initial $r$ and a large initial covariance, then an immediate Kalman update to that measurement.

**Pruning and identity:** components with $r$ below `**pruning_threshold`** are removed. When $r >$ `**existence_threshold**`, `**detection_count` ≥ `min_track_length**`, and no ID yet, a `**track_id**` is assigned (`prune_and_merge` — note: there is **no explicit merge** of duplicate Bernoullis in the base class).

**Confirmed tracks for display/logging** additionally require heuristic **confidence** (`calculate_confidence`), speed bounds, and `**min_display_confidence`** (`get_confirmed_tracks`).

### 3.3 Relation to textbook PMB / PMBM (RFI caveat)

For external accuracy, the following distinctions matter:


| Textbook PMB / PMBM                                              | This implementation                                                                                                                |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Poisson birth field, structured multi-target hypothesis handling | Births are **ad hoc** new Bernoullis from unused detections; `**birth_rate`** is stored but **not** used in the shown update path. |
| Global optimal or marginal association                           | **Greedy** association; measurement used at most once; component **order** matters.                                                |
| Clutter model tied to sensor field of view                       | `**clutter_rate / (128·128)`** is a **fixed normalization**, not automatically scaled to frame resolution.                         |
| Labeled MB / GLMB-style track labels                             | Track IDs are assigned by **heuristic confirmation**, not full labeled RFS machinery.                                              |


**Bottom line:** the tracker is best described as a **PMB-inspired multi-Bernoulli filter with Kalman kinematics and greedy data association**, suitable as an engineering baseline rather than a reference implementation of a specific published PMBM recursion.

### 3.4 Extensions on the PMB path

- **Adaptive RFS family** (`AdaptiveRFSFamilyTracker` in `variants.py`): enriches detections (e.g. bbox/area), **scales** association gates with target speed/size heuristics, and adjusts confirmation thresholds — still the **same single-layer Bernoulli update** as base PMB, not full GLMB/LMB.
- **PMB large/fast** (`PmbLargeFastTracker`): subclasses the adaptive family; **dual soft/hard** masks, **percentile-based** supplemental proposals, stronger births, gentler prune/miss behavior, and **display** smoothing — aimed at large/fast movers.
- **Track-before-detect** (`TrueTbdPmTracker` in `pmb.py`): **no global binary mask** in the loop; builds a **fused soft residual map** ($0.6 \max + 0.4$ mean over a short window), proposes measurements from **local maxima** with adaptive floors, and **scales** $L_{ij}$ by **local integrated evidence** around prediction and detection. Conceptually closer to **TBD-style likelihood lifting** while retaining the same Bernoulli existence update structure.

---

## 4. Advanced baseline and variants (concise)

### 4.1 Advanced baseline

**Class:** `AdvancedSatelliteTracker` in `tracking_core/advanced.py`.  
**CLI:** `run_advanced.py` — **“Advanced baseline”**.

**Pipeline (high level):**

1. **Background** — median (or mode for short clips) over frames.
2. **Foreground** — `absdiff`, threshold, optional morphology.
3. **Detection** — **DBSCAN** on foreground pixels → centroids, bounding boxes, areas.
4. **Association** — cost matrix using **Mahalanobis** / distance with **gating**; **Hungarian algorithm** (`linear_sum_assignment`) for one-to-one matching.
5. **Filtering** — per-track **adaptive 2D Kalman** (`AdaptiveKalmanFilter2D`).
6. **Track quality** — confidence scoring, **motion** filters (e.g. min speed to suppress stars/hot pixels), lifecycle (lost/re-ID timeouts).

**Core equations** match the standard linear Gaussian Kalman cycle (see also `README_Satellite_Tracking.md`):

$$
\hat{\mathbf{x}}*{k|k-1} = \mathbf{F}\hat{\mathbf{x}}*{k-1|k-1}, \quad
\mathbf{P}*{k|k-1} = \mathbf{F}\mathbf{P}*{k-1|k-1}\mathbf{F}^\top + \mathbf{Q},
$$

$$
\mathbf{K}*k = \mathbf{P}*{k|k-1}\mathbf{H}^\top \left(\mathbf{H}\mathbf{P}*{k|k-1}\mathbf{H}^\top + \mathbf{R}\right)^{-1},
\quad
\hat{\mathbf{x}}*{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k(\mathbf{z}*k - \mathbf{H}\hat{\mathbf{x}}*{k|k-1}).
$$

### 4.2 IMM adaptive

**Class:** `IMMAdaptiveMotionTracker`.  
**Idea:** **Heuristic motion modes** (`search`, `cruise`, `maneuver`, `fast`) driven by speed and innovation; each mode scales **Mahalanobis** gates and motion limits. **Not** a full IMM with explicit model probabilities and mixing.

### 4.3 JPDA lite

**Class:** `JPDALiteTracker`.  
**Idea:** Greedy association with optional **soft blending** of neighboring detections (temperature-weighted centroid). **Not** full JPDA joint association probabilities.

### 4.4 MHT lite

**Class:** `MHTLiteTracker`.  
**Idea:** **Hungarian** assignment plus **tentative** handling when two detections are ambiguous (blended update, marked tentative). **Not** an $N$-scan MHT or explicit hypothesis tree.

### 4.5 Particle assisted

**Class:** `ParticleAssistedTracker`.  
**Idea:** Per-track **bootstrap particle filter** blended with Kalman prediction for association costs; resampling against measurements. Detection and overall structure remain **Advanced**-like.

### 4.6 Temporal-accumulation TBD (approx)

**Class:** `TrackBeforeDetectTracker`.  
**Idea:** Overrides preprocessing only: buffers last $W$ **absdiff** maps, fuses $0.6 \max + 0.4$ mean, then **hard threshold** and runs the **standard Advanced** detect-then-track. Documented in-code as **legacy / not classical TBD** on raw images without thresholding.

---

## 5. Registry index (all named algorithms)


| Name (CLI / comparison)            | Kind     | Primary class                  | Notes                                |
| ---------------------------------- | -------- | ------------------------------ | ------------------------------------ |
| Advanced baseline                  | advanced | `AdvancedSatelliteTracker`     | Full `advanced.py` pipeline          |
| Current PMB                        | pmb      | `PoissonMultiBernoulliTracker` | Canonical PMB path, `pmb.py`         |
| PMB large/fast                     | pmb      | `PmbLargeFastTracker`          | Adaptive RFS + large/fast heuristics |
| IMM adaptive                       | advanced | `IMMAdaptiveMotionTracker`     | Heuristic IMM-like gating            |
| JPDA lite                          | advanced | `JPDALiteTracker`              | Soft / greedy JPDA-inspired          |
| MHT lite                           | advanced | `MHTLiteTracker`               | Hungarian + tentative blend          |
| Particle assisted                  | advanced | `ParticleAssistedTracker`      | PF + KF hybrid                       |
| Track-before-detect                | pmb      | `TrueTbdPmTracker`             | Soft map TBD + PMB updates           |
| Temporal-accumulation TBD (approx) | advanced | `TrackBeforeDetectTracker`     | Fused threshold + Advanced           |
| Adaptive RFS family                | pmb      | `AdaptiveRFSFamilyTracker`     | PMB + adaptive gates / confirmation  |


*The comparison runner may list the same logical set; see `TRACKER_VARIANTS` in `tracking_core/variants.py` for the authoritative list and default hyperparameters.*

---

## 6. Qualitative vs measured metrics

In batch comparisons (`run_tracking_comparison.py`), **noise / motion / clutter / compute** scores (1–5) come from **static** per-variant metadata in `TRACKER_VARIANTS`. They are **labels for plotting**, not estimated from a specific clip. **Measured** quantities include runtime, frame count, detection count, and track counts from each run’s `summary.json` (see `README_Tracker_Comparison.md`).

---

## 7. Key file map


| File                                               | Role                                                     |
| -------------------------------------------------- | -------------------------------------------------------- |
| `tracking_core/advanced.py`                        | Advanced pipeline, Kalman, Hungarian, DBSCAN detection   |
| `tracking_core/pmb.py`                             | PMB tracker, `BernoulliComponent`, `TrueTbdPmTracker`    |
| `tracking_core/variants.py`                        | All variant classes, `TRACKER_VARIANTS`, baseline kwargs |
| `tracking_core/tracker_cli.py`                     | CLI, `summary.json`, output paths                        |
| `tracking_core/presets.py`, `tracker_presets.json` | Preset merging for hyperparameters                       |
| `README_Tracker_Comparison.md`                     | Exact code map variant-by-variant                        |
| `README_Satellite_Tracking.md`                     | Advanced pipeline narrative                              |


---

## 8. Summary comparison (operational, not formal optimality)


| Need                                              | Reasonable first try                             |
| ------------------------------------------------- | ------------------------------------------------ |
| Standard detect-then-track with global assignment | **Advanced baseline**                            |
| Variable speed / maneuver heuristics              | **IMM adaptive**                                 |
| Ambiguous dense detections                        | **JPDA lite** / **MHT lite** (understand “lite”) |
| Noisy centroids / streaky appearance              | **Particle assisted**                            |
| Explicit existence probabilities + birth/miss     | **Current PMB** (with §3.3 caveats)              |
| Larger/faster targets, more recall                | **PMB large/fast** or **Adaptive RFS family**    |
| Softer measurement map, TBD-style                 | **Track-before-detect** (`TrueTbdPmTracker`)     |
| Quick temporal fusion baseline                    | **Temporal-accumulation TBD (approx)**           |


---

*Document version: aligned with repository structure as of authoring; implementation details refer to `tracking_core/` sources.*