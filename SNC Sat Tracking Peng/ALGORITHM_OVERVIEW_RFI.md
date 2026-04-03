# Satellite Video Tracking — Algorithm Overview (RFI)

This document describes the **algorithms and methods** implemented in the *SNC Sat Tracking Peng* codebase for passive-optical satellite (or point-source) tracking in video. It is written for **requests for information (RFI)** and external technical review.

**Primary RFI theory anchor:** **PMBM** (`PoissonMultiBernoulliMixtureTracker` in `tracking_core/pmbm.py`) is the main **multi-hypothesis** narrative: a **log-weighted mixture** over several **global association** patterns per frame, built on the same **Bernoulli** measurement update and **Textbook PMB** cost model (**Mahalanobis** gate, frame-scaled $\lambda_c$, Hungarian assignment + **miss** columns, Poisson-style births). **MAP** mixture components drive overlays, `tracks.txt`, and metrics. **Textbook PMB** is the **single-posterior** simplification on the same front end—use it when explaining the core PMB update without mixture branching. Other trackers are **engineering alternatives**; see §5–7.

**Math in this file:** Display and inline formulas use `$...$` and `$$...$$` so they render in **Cursor / VS Code** Markdown preview (enable **Markdown › Math** if needed). Plain **GitHub.com** does not render LaTeX in `.md` files; export to PDF/HTML with a LaTeX-capable tool if you need math there.

---

## 1. Executive summary

The project registers **twelve named tracker configurations** in `tracking_core/variants.py` (`TRACKER_VARIANTS`). For RFI language that reflects **mixture-style** multi-target reasoning on this stack, cite **PMBM** first; use **Textbook PMB** for the **same** detector and Bernoulli formulas **without** retaining multiple association hypotheses; use **Current PMB** and other variants for **tunability, throughput, or heuristics**.

| Role | Family | Core module | One-line idea |
| ---- | ------ | ----------- | ------------- |
| **Primary theory (mixture)** | **PMBM** | `tracking_core/pmbm.py` | Per frame: several **global** assignments (best Hungarian + **forced-miss** perturbations); **log-weights** updated and **normalized**; **cap** hypotheses; **MAP** Bernoulli set for display/logs. Same gated likelihoods, $\lambda_c$, births as Textbook PMB **per** child. |
| **Single-posterior baseline** | **Textbook PMB** | `tracking_core/pmb.py` | One Hungarian solution per frame; no mixture—**simpler** RFI baseline when mixture detail is not needed. |
| **Engineering PMB variants** | PMB path | `pmb.py` + `variants.py` | **Greedy** (**Current PMB**), adaptive gates (**Adaptive RFS**, **PMB large/fast**), **budgeted** likelihoods (**PMB sparse (fast)**), **soft-map** (**Track-before-detect**). |
| **Alternative pipeline** | Advanced | `tracking_core/advanced.py` | DBSCAN + **Hungarian** + explicit tracks + **adaptive 2D Kalman**. **IMM / JPDA / MHT / particle / TBD-approx** are **“lite”**—not full textbook filters unless noted in §6. |

**Outputs** (per run): annotated video, `tracks.txt`, and `summary.json` (see `tracking_core/tracker_cli.py` and `README_Tracker_Comparison.md`).

---

## 2. Terminology (for RFI readers)

- **Implemented algorithm** — Code path executed at runtime (e.g. `PoissonMultiBernoulliMixtureTracker.run`).
- **Variant** — A subclass or configuration that changes part of a base pipeline while reusing the rest.
- **Literature-inspired / “lite”** — Naming aligns with classical methods (JPDA, MHT, IMM, GLMB/LMB) but the code may use **heuristics** or **greedy** steps instead of the full formal filter.
- **MAP (here)** — Highest **log-weight** mixture hypothesis after per-frame normalization; used for video overlay and track log, not a full marginal estimator.

---

## 3. PMBM — primary theory (multi-hypothesis)

**Class:** `PoissonMultiBernoulliMixtureTracker` in `tracking_core/pmbm.py`. **CLI:** **“PMBM”** (`run_pmbm.py`). **Hyperparameters:** `pmbm_k_best` (max assignment variants per parent), `pmbm_max_hypotheses` (mixture cap).

### 3.1 Random-set and mixture view

A **multi-Bernoulli** description uses independent **Bernoulli** components, each with existence $r_i \in [0,1]$ and a spatial density (here **Gaussian** via constant-velocity Kalman on $\mathbf{x} = [x, y, v_x, v_y]^\top$, $dt = 1$).

**Poisson multi-Bernoulli mixture (PMBM)** formulations in the literature represent the multi-target posterior as a **mixture** over **multi-Bernoulli** (and related) structures, with **data association** uncertainty expressed through **multiple weighted hypotheses** rather than a single joint assignment.

This implementation maintains a **finite** list of hypotheses $h = 1,\ldots,H$, each with **log-weight** $\log w_h$ (normalized each frame so $\sum_h w_h = 1$) and its own list of `BernoulliComponent` instances. Each hypothesis is a **full** multi-Bernoulli state for that branch.

### 3.2 Shared front-end and time update (all hypotheses)

**Detection (shared with Textbook / Current PMB):** median background → absdiff / blur / threshold → **DBSCAN** centroids (`compute_background`, `preprocess_frame`, `detect_objects`).

**Predict:** for **every** hypothesis independently, for each component:

$$
\mathbf{x}_{k|k-1} = \mathbf{F}\mathbf{x}_{k-1|k-1}, \qquad
\mathbf{P}_{k|k-1} = \mathbf{F}\mathbf{P}_{k-1|k-1}\mathbf{F}^\top + \mathbf{Q},
\qquad
r_{k|k-1} = p_s \, r_{k-1|k-1},
$$

with $p_s$ = **`survival_prob`**.

### 3.3 Measurement update: cost matrix, branches, weights

For a given hypothesis with $M$ predicted components and $N$ detections, build the **same** rectangular cost matrix as **Textbook PMB**: gated **Mahalanobis** innovations, $C_{ij} = -\log L_{ij}$ for valid $(i,j)$, **miss** columns $C_{i,N+i}$ at fixed cost, large **BIG** fill elsewhere (`_build_textbook_cost_matrix`).

**Association variants:** enumerate up to **`pmbm_k_best`** assignments: (1) global optimum from `linear_sum_assignment`; (2) additional solutions from **forced-miss** perturbations (one row forced to its miss column, then re-solve), **deduplicated** by assignment signature. This is a **practical** multi-solution set—not **Murty’s** full ranked list on the assignment polytope.

For parent log-weight $\log w_p$ and child assignment cost $c$ (sum of selected $C_{ij}$), with $c^\star = \min c$ over siblings from that parent, use

$$
\log w_{\mathrm{child}} = \log w_p - (c - c^\star),
$$

so the **best** child from that parent retains the parent’s relative weight; then apply **log-sum-exp** normalization across **all** children from **all** parents. Keep the top **`pmbm_max_hypotheses`** by weight and renormalize.

**Bernoulli update** for each child: apply the **Textbook PMB** association outcome—associated cells use the same $r_{ij}^{+}$ ratio with $\lambda_c = \texttt{clutter\_rate}/(H\cdot W)$ (or $128^2$ fallback); misses use $r \leftarrow (1-p_d)r$; unused detections spawn births with $r_0 \approx \texttt{birth\_rate}/(\texttt{birth\_rate}+\lambda_c)$.

### 3.4 Output, pruning, and MAP

**Pruning:** each hypothesis drops components with $r$ below **`pruning_threshold`** and very old weak components (same age rule as base PMB). **Track IDs** are assigned only on the **MAP** hypothesis (`prune_and_merge`), then copied back into that slot in the mixture.

**Display / logs:** `get_confirmed_tracks` reads **`self.bernoulli_components`**, which is set to a **deep copy** of the **MAP** hypothesis after each `update_components` and kept consistent after pruning.

### 3.5 Caveat: not full δ-PMBM / labeled RFS

| Literature PMBM / δ-GLMB-style | This **PMBM** implementation |
| ------------------------------ | ----------------------------- |
| Poisson birth **field** over state; full MBM structure | Births remain **new Bernoullis from unused detections** with $r_0$ from **`birth_rate`** vs $\lambda_c$. |
| Systematic **k-best** joint association (e.g. Murty) | **Hungarian + forced-miss** variants only; capped at **`pmbm_k_best`**. |
| Labeled tracks and hypothesis trees | **MAP**-based **track_id** assignment; no labeled RFS recursion. |
| Exact mixture reduction / merging theory | **Top-$H$** pruning by weight; no principled merge of duplicate MB structures beyond deduplication of assignment keys. |

**Bottom line for RFI:** **PMBM** here is a **defensible engineering mixture** over **global association** patterns on the **Textbook PMB** measurement model—closer to PMBM narrative than a single posterior, but **not** a reference δ-PMBM filter from the literature.

---

## 4. Textbook PMB — single-posterior baseline (short)

**Class:** `TextbookPoissonMultiBernoulliTracker` in `tracking_core/pmb.py`. **CLI:** **“Textbook PMB”**.

Same **§3.2** front-end and **§3.3** Bernoulli / $\lambda_c$ / birth formulas, but **one** Hungarian solution per frame—**no** mixture. Use this when the RFI should stay **minimal** while still citing **Mahalanobis** gating, **frame-scaled** clutter, and **Poisson-style** births.

---

## 5. Current PMB and other PMB-path variants (short)

**Current PMB:** **greedy** association, **physical** gates, clutter often $128^2$-normalized; births not tied to **`birth_rate`** the same way.

| Name | Essence |
| ---- | ------- |
| **Adaptive RFS family** | Richer detections; scaled gates—still **single-layer** Bernoulli update. |
| **PMB large/fast** | Adaptive RFS + masks / proposals / display smoothing. |
| **PMB sparse (fast)** | Greedy PMB + spatial gating, budgets, streaming. |
| **Track-before-detect** | Soft residual map + local evidence scaling. |

---

## 6. Advanced baseline and “lite” variants (short)

**Advanced baseline:** median background → DBSCAN → Hungarian → **adaptive 2D Kalman** per track.

**IMM / JPDA / MHT / particle / TBD-approx:** heuristic or partial implementations—see `README_Tracker_Comparison.md`.

---

## 7. Registry index (all named algorithms)


| Name (CLI / comparison)            | Kind     | Primary class                  | Notes                                |
| ---------------------------------- | -------- | ------------------------------ | ------------------------------------ |
| PMBM                               | pmb      | `PoissonMultiBernoulliMixtureTracker` | **RFI primary theory** (§3), MAP output |
| Textbook PMB                       | pmb      | `TextbookPoissonMultiBernoulliTracker` | Single-posterior reference (§4)      |
| Current PMB                        | pmb      | `PoissonMultiBernoulliTracker` | Greedy PMB, `pmb.py`                 |
| PMB large/fast                     | pmb      | `PmbLargeFastTracker`          | Adaptive RFS + large/fast heuristics |
| PMB sparse (fast)                  | pmb      | `SparseBudgetPmTracker`        | Streaming + gated likelihoods + budgets |
| Track-before-detect                | pmb      | `TrueTbdPmTracker`             | Soft map TBD + PMB updates           |
| Adaptive RFS family                | pmb      | `AdaptiveRFSFamilyTracker`     | PMB + adaptive gates / confirmation  |
| Advanced baseline                  | advanced | `AdvancedSatelliteTracker`     | Full `advanced.py` pipeline          |
| IMM adaptive                       | advanced | `IMMAdaptiveMotionTracker`     | Heuristic IMM-like gating            |
| JPDA lite                          | advanced | `JPDALiteTracker`              | Soft / greedy JPDA-inspired          |
| MHT lite                           | advanced | `MHTLiteTracker`               | Hungarian + tentative blend          |
| Particle assisted                  | advanced | `ParticleAssistedTracker`      | PF + KF hybrid                       |
| Temporal-accumulation TBD (approx) | advanced | `TrackBeforeDetectTracker`     | Fused threshold + Advanced           |


*Authoritative kwargs and list: `TRACKER_VARIANTS` in `tracking_core/variants.py`.*

---

## 8. Performance expectations (no ground truth)

**Classical empirical rates are undefined without labels.** Detection rate (recall), false-alarm rate, and missed-detection rate require ground truth (or synthetic truth) to classify measurements and tracks as true vs. spurious. Per-run `summary.json` counts (e.g. total detections) are **operational**, not $P_d$, $P_{\mathrm{fa}}$, or $P_{\mathrm{miss}}$.

**Footnote (applies to both tables below):** *Empirical $P_d$, $P_{\mathrm{fa}}$, and missed-detection rates are **not reported**: there is no labeled evaluation set on the project videos. Table 8.1 entries are **engineering expectations** from architecture and registry defaults/overrides in `tracking_core/variants.py`, not validated performance. Table 8.2 lists **filter tuning parameters** for the Bernoulli measurement model where applicable; they are **not** the empirical hit rate of the threshold + DBSCAN front-end.*

**Trend reference:** *Higher / similar / lower* are **relative** to the usual default in each pipeline family: **Advanced baseline** for `advanced` trackers, **Current PMB** for `pmb` trackers.

### Table 8.1 — Qualitative trends (all registered algorithms)

| Algorithm | Expected detection / recall (trend) | Expected false-alarm & clutter sensitivity (trend) | Expected missed detection & track drop-out (trend) |
| --------- | ------------------------------------- | ---------------------------------------------------- | --------------------------------------------------- |
| Advanced baseline | Reference | Reference | Reference |
| Current PMB | Reference | Reference | Reference |
| Textbook PMB | Similar (same front-end); global Hungarian + Mahalanobis gate can pass marginal hits differently than greedy PMB | Similar clutter model ($\lambda_c$, births); registry lowers `min_display_confidence` vs default PMB, so **more** weak tracks may appear | Similar miss update $(1-p_d)r$; association differs from greedy path—drop-out **similar**, fragmentation pattern may differ |
| PMBM | Similar to Textbook PMB (same cost model per hypothesis) | Similar to Textbook PMB | **Similar or slightly better** under association ambiguity (mixture defers to MAP); not a guarantee of lower FA |
| PMB large/fast | **Higher** (softer / OR masks, supplemental percentile blobs, larger `max_detection_area`) | **Higher** (more proposals → more spurious tracks unless pruned) | **Lower** tendency to lose established targets (gentler miss / display grace, tuned `detection_prob` and clutter) |
| PMB sparse (fast) | **Lower** ceiling in dense scenes (per-frame detection and component **caps**, spatial gate on likelihoods) | **Lower** from hard budgets and gating (fewer simultaneous hypotheses) | **Higher** risk that real targets fall outside budget or gate and **drop** |
| Track-before-detect | **Different** failure mode vs global binary mask (soft fused map, local evidence); can surface weak peaks but **capped** proposals/frame | **Medium** (percentile floor and patch-based scaling; not monotonically “safer” than threshold PMB) | **Medium** (strict `min_display_confidence` in registry; TBD-style birth/update trade-offs) |
| Adaptive RFS family | **Higher** than Current PMB (richer `detect_objects`, bbox/area, scaled gates) | **Higher** clutter exposure vs tighter Current PMB | **Lower** drop-out for large/fast movers via dynamic confirmation and gates (still single-layer PMB update) |
| IMM adaptive | **Higher** effective linkage under **variable speed** (scaled Mahalanobis / motion gates per mode) | **Similar** front-end; clutter score is a plot label, not measured | **Lower** tendency to **miss** associations during maneuvers vs baseline Advanced |
| JPDA lite | Similar front-end | **Higher** tolerance when multiple detections compete (soft / neighbor-aware association—**lite**, not full JPDA) | Similar timeouts vs baseline; ambiguous cases may **linger** rather than hard-split |
| MHT lite | Similar front-end | **Higher** tolerance under ambiguity (tentative blend + longer windows in registry) | Similar; tentative tracks can **delay** commitment |
| Particle assisted | Similar detections | Similar | **Lower** drop-out from **noisy centroids** (particles stabilize state vs Kalman-only in places) |
| Temporal-accumulation TBD (approx) | **Higher** proposal rate (lower `bg_threshold`, lower `min_intensity` in registry) | **Higher** FA risk from aggressive thresholding after temporal fusion | Depends on fusion and gates; **similar** Advanced association afterward |

### Table 8.2 — Bernoulli filter parameters (PMB path only)

These values are **`PMB_BASELINE_KWARGS` merged with each entry’s `overrides`** in `tracking_core/variants.py`. $\lambda_c$ is **frame-scaled** as $ \texttt{clutter\_rate}/(H{\cdot}W) $ for **Textbook PMB**, **PMBM**, **Track-before-detect**, and when **`pmb_frame_scaled_clutter`** is enabled (**PMB sparse (fast)**); **Current PMB** and some greedy paths may use the implementation’s $128^2$ normalization—see `pmb.py` and `README_Tracker_Comparison.md`.

| Algorithm | `detection_prob` (modeled $p_d$ in Bernoulli update) | `clutter_rate` (drives $\lambda_c$) | `birth_rate` |
| --------- | ---------------------------------------------------- | ------------------------------------- | ------------ |
| Advanced baseline | — | — | — |
| Current PMB | 0.85 | 5.0 | 0.1 |
| Textbook PMB | 0.85 | 5.0 | 0.1 |
| PMBM | 0.85 | 5.0 | 0.1 |
| PMB large/fast | 0.82 | 6.0 | 0.1 |
| PMB sparse (fast) | 0.85 | 5.0 | 0.1 |
| Track-before-detect | 0.80 | 5.5 | 0.1 |
| Adaptive RFS family | 0.85 | 5.0 | 0.1 |
| IMM adaptive | — | — | — |
| JPDA lite | — | — | — |
| MHT lite | — | — | — |
| Particle assisted | — | — | — |
| Temporal-accumulation TBD (approx) | — | — | — |

*Table 8.2 footnote: parameters are **not** ROC operating points. Advanced trackers use thresholds such as `bg_threshold`, `mahalanobis_threshold`, and `min_detection_confidence` from `ADVANCED_BASELINE_KWARGS` plus overrides instead of `detection_prob` / `clutter_rate` / `birth_rate`.*

**Paths to empirical metrics later:** label a subset of frames, run synthetic video with known trajectories, or define **diagnostic** proxies (unmatched detections per frame, fragmentation)—proxies still require careful definition and are not classical $P_{\mathrm{fa}}$ without truth.

---

## 9. Qualitative vs measured metrics

In batch comparisons (`run_tracking_comparison.py`), **noise / motion / clutter / compute** scores (1–5) are **static** labels in `TRACKER_VARIANTS`, not estimated per clip. **Measured** fields include runtime, frame count, detection count, and track counts from each `summary.json` (see `README_Tracker_Comparison.md`).

---

## 10. Key file map


| File                                               | Role                                                     |
| -------------------------------------------------- | -------------------------------------------------------- |
| `tracking_core/pmbm.py`                            | **PMBM** mixture tracker                                 |
| `tracking_core/pmb.py`                             | Bernoulli PMB, **Textbook PMB**, sparse/TBD variants     |
| `tracking_core/advanced.py`                        | Advanced pipeline, Kalman, Hungarian, DBSCAN           |
| `tracking_core/variants.py`                        | Variant classes, `TRACKER_VARIANTS`, baseline kwargs   |
| `tracking_core/tracker_cli.py`                     | CLI, `summary.json`, output paths                      |
| `tracking_core/presets.py`, `tracker_presets.json` | Preset merging                                         |
| `README_Tracker_Comparison.md`                     | Exact code map variant-by-variant                      |
| `README_Satellite_Tracking.md`                     | Advanced pipeline narrative                            |


---

## 11. Summary comparison (operational, not formal optimality)


| Need                                              | Reasonable first try                             |
| ------------------------------------------------- | ------------------------------------------------ |
| **RFI theory: mixture over association**          | **PMBM** (§3)                                    |
| **RFI theory: single posterior, same measurements** | **Textbook PMB** (§4)                          |
| Simple greedy PMB                                 | **Current PMB**                                  |
| Standard detect-then-track with explicit tracks   | **Advanced baseline**                            |
| Variable speed / maneuver heuristics              | **IMM adaptive**                                 |
| Ambiguous dense detections                        | **JPDA lite** / **MHT lite** (understand “lite”) |
| Noisy centroids / streaky appearance              | **Particle assisted**                            |
| Larger/faster targets, more recall                | **PMB large/fast** or **Adaptive RFS family**    |
| Throughput / sparse scoring                       | **PMB sparse (fast)**                            |
| Softer measurement map, TBD-style                 | **Track-before-detect**                          |
| Quick temporal fusion baseline                    | **Temporal-accumulation TBD (approx)**           |


---

*Document version: aligned with `tracking_core/` sources; for air-to-air framing see [AIR_TO_AIR_ADAPTATION.md](AIR_TO_AIR_ADAPTATION.md).*
