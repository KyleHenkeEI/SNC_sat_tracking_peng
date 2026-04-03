# Air-to-air passive IR adaptation

This note describes how the existing passive IR tracker stack can be adapted into an **air-to-air** capability. The emphasis is on the **future adaptation path**: what carries over from the current tracker baseline, what must change for airborne use, what technical risks dominate, and how the capability would be matured on representative data for an RFI or follow-on prototype effort.

**Algorithm baseline for RFI:** treat **PMBM** (`run_pmbm.py`, `PoissonMultiBernoulliMixtureTracker`) as the primary **theory** reference—a **log-weighted mixture** over several **global association** patterns each frame, built on the same detect-then-track front end, **Mahalanobis** gating, **frame-scaled** clutter, **Hungarian**-style costs, and **Poisson-style** births as in **Textbook PMB**, with **MAP** output for overlays and logs. For **early** air-to-air demos or lower CPU budgets, **Textbook PMB** (`run_textbook_pmb.py`) is the **single-posterior** simplification on identical measurements. Airborne maturity remains dominated by **motion compensation**, **timing**, **calibration**, and **validation**, not by renaming trackers. **Current PMB**, **PMB sparse (fast)**, **Advanced**, and other entries remain **secondary** for throughput, heuristics, or comparison; see [ALGORITHM_OVERVIEW_RFI.md](ALGORITHM_OVERVIEW_RFI.md).

The current codebase should be viewed as a **reusable starting point**, not a complete operational air-to-air solution. Its value is that it already provides detection, tracking, configuration, and evaluation infrastructure that can be extended rather than replaced.

---

## 1. What transfers directly from the current stack

Several elements of the current passive tracker architecture are still useful in air-to-air adaptation:

- **PMBM track layer**: multiple hypotheses, each a multi-Bernoulli set with **existence** $r$, **gated** likelihoods, and **global** assignment variants per frame; **normalized log-weights** and a **MAP** branch for external-facing outputs—this is the lead **theory** story for RFI on this codebase.
- **Textbook PMB (simpler path)**: same **clutter/birth** scaling and **Hungarian** assignment as one hypothesis only—useful when mixture branching is not yet needed or for lighter runs.
- **Shared detection front-end**: median background, absdiff/threshold preprocessing, and **DBSCAN**-clustered centroids—the same measurements feed **PMBM**, **Textbook PMB**, and **Current PMB**; retuning thresholds and detector knobs is the first line of adaptation on new imagery.
- **Configuration and presets**: `tracker_presets.json` and variant kwargs allow scenario-specific tuning without rewriting the tracker core.
- **Batch evaluation**: run scripts and `summary.json` / comparison tooling support repeatable experiments as airborne clips arrive.

**Other trackers (brief):** **Current PMB** (greedy association, physical motion gates) and **PMB sparse (fast)** (throughput-oriented) stay on the same Bernoulli family without mixture; **Advanced** and **“lite”** variants are a **separate** detect-then-track pipeline—use them for Kalman-centric or legacy comparisons, not as the primary **PMBM** theory reference.

In RFI terms, the adaptation path is best framed as **extending a PMBM-centered passive tracking baseline** (with **Textbook PMB** as the optional single-posterior stepping stone), plus motion integration—not as creating a new tracker from scratch.

---

## 2. What changes in air-to-air tracking

Air-to-air passive IR changes the operating assumptions in ways that materially affect both detection and tracking:

- **Ownship motion matters**: apparent target motion is driven by both target kinematics and host aircraft motion, so background motion can no longer be treated as quasi-static.
- **Line-of-sight rates can be much higher**: crossing targets, turns, and narrow-FOV sensors can produce rapid focal-plane motion and abrupt direction changes.
- **Backgrounds are more complex**: clouds, horizon, terrain, sun glint, jet wash, and plume structure can all create clutter-like signatures or invalidate simple background models.
- **Target signature is aspect-dependent**: passive IR appearance may shift between point-like, plume-dominant, or partially extended signatures depending on geometry, propulsion state, and atmosphere.
- **Airborne imagery is integration-dependent**: performance depends strongly on stabilization quality, timing integrity, calibration, and mounting geometry.

For that reason, the central air-to-air problem is not just tracker tuning. It is the combination of **motion compensation**, **sensor integration**, **calibration**, and **validation** around the tracker.

---

## 3. Recommended adaptation strategy

The recommended path is incremental: start with the fastest route to representative feasibility, then add the integration layers needed for operational robustness.

### 3.1 Phase 1: Stabilized or approximately stabilized imagery

The fastest route to early air-to-air feasibility is to operate on imagery that is already **line-of-sight stabilized** by hardware or exported in a stabilized view. This preserves the closest match to the current background-subtraction and detect-then-track assumptions.

If fully stabilized imagery is not yet available, a software-only image-registration step can still support early feasibility studies:

- **Translation-only phase correlation** is useful as a **prototype bridge** for offline analysis and parameter tuning.
- It can help determine whether target signatures are detectable and whether the tracking logic is directionally suitable on airborne clips.
- It should **not** be treated as the endpoint for production air-to-air processing, because rotation, parallax, vibration, and low-texture scenes will eventually break a pure translation model.

The right RFI framing is therefore:

- **Near-term**: use stabilized imagery where possible; otherwise use lightweight image alignment to bootstrap feasibility.
- **Mid-term**: replace or augment this with higher-fidelity airborne motion compensation.

### 3.2 Phase 2: Motion compensation from aircraft metadata

The next major adaptation step is to fuse video with **IMU / INS / gimbal** information:

- Align aircraft and sensor metadata to frame capture time.
- Use camera intrinsics and extrinsics to convert measured host motion into image-domain or LOS-domain compensation.
- Characterize buffering and latency so compensation is tied to **exposure time**, not just software arrival time.
- Validate axis conventions and boresight alignment for the installed sensor.

This is the key transition from a vision-only prototype to a fieldable airborne architecture. It also reduces dependence on scene texture, which is important for passive IR.

### 3.3 Phase 3: Higher-fidelity scene motion and background handling

Once synchronized metadata exists, the tracker can be extended beyond simple translation assumptions:

- **Richer warp models** such as affine or homography alignment can improve robustness under roll and moderate viewpoint change.
- **Hybrid inertial-image registration** can reduce drift while still using image evidence.
- **Adaptive background models** can replace a single global clip median when the scene evolves over time or when clouds and horizon structure dominate.
- **Mode-dependent detection logic** can be introduced for point targets, plume-dominant signatures, or horizon-clutter scenes.

This is also the point where the architecture can branch depending on intended operational use: narrow-FOV track refinement, wider-FOV search, stabilized turret video, or fixed-mount body-camera operation.

### 3.4 Phase 4: Operational maturation

Before claiming operational relevance, the system should be validated across representative mission geometries:

- head-on, tail chase, and crossing engagements,
- low and high ownship maneuver,
- clear air, cloud background, and horizon crossings,
- weak point-source targets versus stronger plume-dominant signatures,
- stabilized and unstabilized collection modes.

The emphasis at this stage is not only raw detection rate, but also track continuity, false-track behavior, latency, and robustness to integration error.

---

## 4. Key technical work packages

An RFI-oriented adaptation plan is best described as a set of work packages rather than a list of files or code hooks.

### 4.1 Sensor and platform integration

Required future work:

- define the camera, IMU, INS, and gimbal interfaces,
- establish a common timing source,
- measure capture-to-processing latency,
- document installation geometry and sensor mounting assumptions.

This work determines whether airborne motion can be compensated consistently.

### 4.2 Calibration and boresight

Required future work:

- camera intrinsics (focal length, principal point, distortion),
- camera-to-body or camera-to-gimbal extrinsics,
- sign conventions and axis mapping,
- maintenance procedures for boresight drift or remount events.

Without this layer, inertial compensation can be misleading or harmful.

### 4.3 Detection and clutter adaptation

Required future work:

- retune thresholds and clutter assumptions for clouds, horizon, and terrain backgrounds,
- support target signatures that change with aspect and propulsion state,
- evaluate whether region-dependent or adaptive thresholds are needed,
- refine target-size assumptions for narrow-FOV versus wider-FOV collection.

### 4.4 Track-model adaptation

Required future work:

- retune **PMBM** / **Textbook PMB** parameters for higher LOS rates and clutter: **Mahalanobis** gate threshold, **`clutter_rate`** (per-pixel intensity), **`birth_rate`**, **`detection_prob`** / **`survival_prob`**, confirmation/pruning thresholds, and for **PMBM** specifically **`pmbm_k_best`** and **`pmbm_max_hypotheses`** (mixture branching vs CPU),
- preserve continuity through intermittent low contrast or partial target loss,
- retune track initiation, confirmation, and pruning for airborne false-alarm regimes,
- consider mode-dependent kinematics if mission use cases span search, pursuit, and crossing geometries.

**Current PMB** adds **physical** speed/accel/turn gates that **PMBM** / **Textbook PMB** omit; if those heuristics help unstable airborne clips, treat them as an **engineering** overlay, not a replacement for the RFI theory anchor above.

---

## 5. Principal risks and recommended mitigations

### 5.1 Timing error

**Risk:** Even small timing offsets between video and inertial data can degrade compensation and destabilize tracks.

**Mitigation:** Use a common clock domain, align to capture time or mid-exposure time, and explicitly measure end-to-end latency.

### 5.2 Calibration error

**Risk:** Incorrect extrinsics, boresight, or sign conventions can make motion compensation worse than no compensation.

**Mitigation:** Include calibration and installation validation as a formal work package, with repeatable acceptance checks after sensor integration.

### 5.3 Scene-driven clutter

**Risk:** Clouds, horizon edges, sun glint, hot terrain, and plume structure may increase false detections or confuse target/background separation.

**Mitigation:** Build a representative dataset early and tune by scenario class rather than assuming one global parameter set.

### 5.4 Motion-model mismatch

**Risk:** Simplified motion assumptions may fail under rapid LOS rate, strong turn dynamics, or viewpoint-induced signature change.

**Mitigation:** Use staged upgrades from broad kinematic retuning to richer compensation and, if needed, multiple motion modes.

### 5.5 Limited representativeness of early data

**Risk:** Success on a small number of clips may not generalize to operational collection conditions.

**Mitigation:** Construct a scenario matrix that spans sensor modes, engagement geometries, backgrounds, and target signature conditions.

---

## 6. Validation approach for an air-to-air program

The adaptation effort should be coupled to a structured validation plan. Recommended elements:

- **Representative data collection**: collect or curate clips spanning clear air, cloud background, horizon crossings, low and high maneuver, and multiple target aspects.
- **Metadata capture**: retain frame timing, frame rate, sensor mode, host motion, and any available aircraft or gimbal state.
- **Truthing strategy**: where precise truth is unavailable, use synchronized aircraft state, event logs, or manual annotation on selected clips to establish performance trends.
- **Performance measures**: evaluate track initiation, continuity, false-track rate, localization consistency, latency, and sensitivity to maneuver and background class.
- **Ablation testing**: compare stabilized-only, metadata-assisted, and higher-order compensation paths to quantify the value of each added integration layer.

For RFI purposes, this validation story is as important as the algorithm description, because it explains how risk would be retired in a follow-on effort.

---

## 7. Recommended initial demonstration path

If asked for the most practical way to demonstrate near-term air-to-air relevance, the recommended sequence is:

1. Start with **stabilized** or approximately stabilized passive IR clips and tune **PMBM** (or **Textbook PMB** for a lighter first pass) plus the shared detection front-end; optionally compare **Current PMB** or **PMB sparse (fast)** for throughput or greedy behavior.
2. Add synchronized **IMU / INS / gimbal** metadata and validate timing and calibration conventions.
3. Introduce richer motion compensation and adaptive background handling where airborne scene motion breaks simple assumptions.
4. Evaluate across a scenario matrix broad enough to show both capability and limitations.

This path minimizes early integration risk while preserving a credible route to operational maturity.

---

## 8. Current code as a prototype enabler

The repository includes optional image preprocessing (including air-to-air-oriented hooks where enabled), **`tracker_presets.json`**, **`run_pmbm.py`** as the primary **RFI theory** entry point (**mixture** over associations), and **`run_textbook_pmb.py`** as the **single-posterior** baseline on the same measurements. **Current PMB**, **PMB large/fast**, **PMB sparse (fast)**, **Track-before-detect**, and the **Advanced** family remain available for **comparison, recall, or throughput** experiments.

These assets are **enablers for experimentation and transition**, not proof that the full airborne integration problem is solved.

For full theory, equations, and implementation caveats (including the distinction from full **δ-PMBM** / labeled RFS in the literature), see [ALGORITHM_OVERVIEW_RFI.md](ALGORITHM_OVERVIEW_RFI.md).
