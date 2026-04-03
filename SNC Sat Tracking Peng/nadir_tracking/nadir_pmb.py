"""
NadirPMBTracker – PMB tracking for space-looking-down (LEO/MEO nadir) video.

Key design decisions
--------------------
* **Phase-correlation registration** — uses the whole image spectrum instead of
  point features, so bright targets cannot corrupt the ego-motion estimate
  (ORB keypoints on a missile plume bias the homography, cancelling the target
  in the residual).
* **SNR + absolute peak detection** with edge exclusion.
* **Coasting** — established tracks hold through missed-detection gaps.
* **Raw-centroid display** — the overlay marker sits on the last measurement,
  not the Kalman-smoothed position, eliminating visual centroid lag.
"""
from __future__ import annotations

import time
from collections import deque
from copy import deepcopy

import cv2
import numpy as np
from scipy.ndimage import maximum_filter
from sklearn.cluster import DBSCAN

from tracking_core.pmb import BernoulliComponent, PoissonMultiBernoulliTracker

from nadir_tracking.ego_motion import (
    motion_compensated_residual,
    multi_frame_residual,
)


def _trajectory_linearity(comp) -> float:
    """Net displacement / total path length.  ~1.0 for straight, ~0 for zig-zag."""
    pts = list(comp.history)
    if len(pts) < 2:
        return 1.0
    total_path = 0.0
    for i in range(1, len(pts)):
        total_path += np.linalg.norm(pts[i] - pts[i - 1])
    if total_path < 1e-3:
        return 0.0
    net = np.linalg.norm(pts[-1] - pts[0])
    return net / total_path


class NadirPMBTracker(PoissonMultiBernoulliTracker):
    """PMB tracker for nadir Earth-observation with moving backgrounds."""

    def __init__(
        self,
        input_video_path: str,
        output_video_path: str,
        *,
        # --- ego-motion ---
        registration_method: str = "phase",
        orb_max_features: int = 1000,
        orb_good_ratio: float = 0.75,
        residual_stack_depth: int = 1,
        # --- peak-finding detection ---
        snr_threshold: float = 4.0,
        local_mean_kernel: int = 51,
        peak_min_distance: int = 12,
        peak_min_residual: int = 8,
        peak_max_detections: int = 80,
        bright_abs_threshold: int = 35,
        # --- coasting ---
        coast_frames: int = 20,
        coast_existence_floor: float = 0.55,
        **kwargs,
    ):
        self.registration_method = registration_method
        self.orb_max_features = int(orb_max_features)
        self.orb_good_ratio = float(orb_good_ratio)
        self.residual_stack_depth = max(1, int(residual_stack_depth))

        self.snr_threshold = float(snr_threshold)
        self.local_mean_kernel = int(local_mean_kernel) | 1
        self.peak_min_distance = int(peak_min_distance)
        self.peak_min_residual = int(peak_min_residual)
        self.peak_max_detections = int(peak_max_detections)
        self.bright_abs_threshold = int(bright_abs_threshold)

        self.coast_frames = int(coast_frames)
        self.coast_existence_floor = float(coast_existence_floor)

        self._prev_frames: deque[np.ndarray] = deque(maxlen=max(2, self.residual_stack_depth + 1))
        self._frame_idx = 0
        self._registration_stats: list[dict] = []

        super().__init__(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            **kwargs,
        )

    # ------------------------------------------------------------------
    def compute_background(self, frames):
        if frames:
            self.background = np.zeros_like(frames[0], dtype=np.uint8)
        return self.background

    # ------------------------------------------------------------------
    # Preprocess: ego-motion-compensated residual + normalized SNR map
    # ------------------------------------------------------------------
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        self._prev_frames.append(frame.copy())

        if len(self._prev_frames) < 2:
            self._last_residual = np.zeros_like(frame, dtype=np.uint8)
            self._last_snr = np.zeros_like(frame, dtype=np.float32)
            return np.zeros_like(frame, dtype=np.uint8)

        frames_list = list(self._prev_frames)

        # Always use single-pair residual (more reliable than multi-frame)
        residual, H, n_inliers = motion_compensated_residual(
            frames_list[-2], frames_list[-1],
            method=self.registration_method,
            max_features=self.orb_max_features,
            good_ratio=self.orb_good_ratio,
        )
        self._registration_stats.append({"inliers": n_inliers, "method": self.registration_method})

        self._last_residual = residual

        # Build brightness-normalized SNR map
        res_f = cv2.GaussianBlur(residual, (3, 3), 0.7).astype(np.float32)
        k = self.local_mean_kernel
        local_mean = cv2.GaussianBlur(res_f, (k, k), 0)
        local_sq_mean = cv2.GaussianBlur(res_f ** 2, (k, k), 0)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0.01))
        snr_map = (res_f - local_mean) / local_std
        self._last_snr = snr_map

        binary = (snr_map > self.snr_threshold).astype(np.uint8) * 255
        return binary

    # ------------------------------------------------------------------
    # Detect: local-maxima peak finding
    # ------------------------------------------------------------------
    def detect_objects(self, binary_frame: np.ndarray, original_frame: np.ndarray) -> list[dict]:
        snr_map = getattr(self, "_last_snr", None)
        residual = getattr(self, "_last_residual", original_frame)
        if snr_map is None:
            return []

        bright_abs = self.bright_abs_threshold
        footprint_size = max(3, self.peak_min_distance)

        # SNR-based peaks
        snr_local_max = maximum_filter(snr_map, size=footprint_size)
        snr_peaks = (snr_map == snr_local_max) & (snr_map >= self.snr_threshold)

        # Absolute-residual-based peaks
        res_f = residual.astype(np.float32)
        res_local_max = maximum_filter(res_f, size=footprint_size)
        abs_peaks = (res_f == res_local_max) & (residual >= bright_abs)

        peaks = (snr_peaks | abs_peaks) & (residual >= self.peak_min_residual)

        # Edge exclusion
        edge = 4
        h_img, w_img = residual.shape[:2]
        peaks[:edge, :] = False
        peaks[-edge:, :] = False
        peaks[:, :edge] = False
        peaks[:, -edge:] = False

        peak_ys, peak_xs = np.where(peaks)
        if len(peak_ys) == 0:
            return []

        snr_vals = snr_map[peak_ys, peak_xs]
        order = np.argsort(snr_vals)[::-1]
        peak_ys = peak_ys[order[:self.peak_max_detections]]
        peak_xs = peak_xs[order[:self.peak_max_detections]]

        if len(peak_ys) == 1:
            x, y = int(peak_xs[0]), int(peak_ys[0])
            intensity = float(residual[y, x])
            return [{
                "centroid": (x, y),
                "bbox": (x - 2, y - 2, 5, 5),
                "area": 1,
                "intensity": intensity,
            }]

        coords = np.column_stack((peak_xs, peak_ys))
        clustering = DBSCAN(eps=self.peak_min_distance, min_samples=1).fit(coords)

        detections = []
        for cid in set(clustering.labels_):
            if cid == -1:
                continue
            mask = clustering.labels_ == cid
            pts = coords[mask]
            weights = snr_map[pts[:, 1], pts[:, 0]]
            w_sum = weights.sum() + 1e-10
            cx = int(np.round(np.sum(pts[:, 0] * weights) / w_sum))
            cy = int(np.round(np.sum(pts[:, 1] * weights) / w_sum))
            h, w = residual.shape[:2]
            cx = min(max(cx, 0), w - 1)
            cy = min(max(cy, 0), h - 1)
            intensity = float(residual[cy, cx])
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            detections.append({
                "centroid": (cx, cy),
                "bbox": (int(x_min) - 1, int(y_min) - 1,
                         int(x_max - x_min) + 3, int(y_max - y_min) + 3),
                "area": len(pts),
                "intensity": intensity,
            })

        return detections

    # ------------------------------------------------------------------
    # Coasting: established tracks decay slowly
    # ------------------------------------------------------------------
    def predict_components(self):
        for comp in self.bernoulli_components:
            comp.predict()
            base_r = self.survival_prob * comp.r
            if (comp.track_id is not None
                    and comp.detection_count >= self.min_track_length
                    and comp.missed_detections <= self.coast_frames):
                comp.r = max(self.coast_existence_floor, base_r)
            else:
                comp.r = base_r

    # ------------------------------------------------------------------
    # Update with spatially-varying clutter
    # ------------------------------------------------------------------
    def update_components(self, detections):
        residual = getattr(self, "_last_residual", None)
        n_components = len(self.bernoulli_components)
        n_detections = len(detections)

        if n_components == 0:
            for det in detections:
                pos = det["centroid"]
                state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
                P = np.eye(4, dtype=np.float64) * 100.0
                comp = BernoulliComponent(0.1, state, P, track_id=None,
                                          process_noise_q=self.process_noise_q)
                self.bernoulli_components.append(comp)
            return

        if n_detections == 0:
            for comp in self.bernoulli_components:
                is_coasting = (comp.track_id is not None
                               and comp.detection_count >= self.min_track_length
                               and comp.missed_detections <= self.coast_frames)
                if is_coasting:
                    comp.r = max(self.coast_existence_floor, comp.r * 0.97)
                else:
                    comp.r = (1 - self.detection_prob) * comp.r
                comp.consecutive_detections = 0
            return

        likelihood_matrix = np.zeros((n_components, n_detections))
        valid_assoc = np.ones((n_components, n_detections), dtype=bool)
        is_coasting_arr = np.zeros(n_components, dtype=bool)

        for i, comp in enumerate(self.bernoulli_components):
            is_coasting_arr[i] = (comp.track_id is not None
                                  and comp.detection_count >= self.min_track_length
                                  and comp.missed_detections <= self.coast_frames)
            for j, det in enumerate(detections):
                if is_coasting_arr[i]:
                    # Coasting: relaxed physical constraints + pixel distance gate
                    pos = np.array(det["centroid"], dtype=np.float64)
                    pred = comp.state[:2]
                    dist = float(np.linalg.norm(pos - pred))
                    est_speed = float(np.linalg.norm(comp.state[2:4]))
                    max_coast_dist = max(30.0, (est_speed + 5.0) * comp.missed_detections)
                    if dist <= max_coast_dist:
                        likelihood_matrix[i, j] = comp.likelihood(det["centroid"])
                    else:
                        valid_assoc[i, j] = False
                        likelihood_matrix[i, j] = 1e-10
                elif comp.check_physical_constraints(
                    det["centroid"],
                    self.max_acceleration,
                    self.max_direction_change,
                    self.max_speed,
                ):
                    likelihood_matrix[i, j] = comp.likelihood(det["centroid"])
                else:
                    valid_assoc[i, j] = False
                    likelihood_matrix[i, j] = 1e-10

        # --- Global Nearest Neighbor (GNN) assignment ---
        # Score = likelihood * r * detection_bonus (tracks with IDs get priority)
        score_matrix = np.full((n_components, n_detections), -np.inf)
        for i in range(n_components):
            comp = self.bernoulli_components[i]
            track_bonus = 10.0 if comp.track_id is not None else 1.0
            for j in range(n_detections):
                if valid_assoc[i, j]:
                    score_matrix[i, j] = likelihood_matrix[i, j] * comp.r * track_bonus

        # Greedy GNN: repeatedly pick the (i,j) pair with highest score
        assigned_comp = {}   # comp_idx -> det_idx
        assigned_det = set() # det_idx set
        sm = score_matrix.copy()
        while True:
            idx = np.unravel_index(np.argmax(sm), sm.shape)
            if sm[idx] <= 0:
                break
            i_best, j_best = idx
            assigned_comp[i_best] = j_best
            assigned_det.add(j_best)
            sm[i_best, :] = -np.inf
            sm[:, j_best] = -np.inf

        updated = []
        used = set(assigned_det)

        for i, comp in enumerate(self.bernoulli_components):
            is_coasting = (comp.track_id is not None
                           and comp.detection_count >= self.min_track_length
                           and comp.missed_detections <= self.coast_frames)

            if i in assigned_comp:
                j = assigned_comp[i]
                det = detections[j]
                if is_coasting:
                    r_new = max(comp.r, self.coast_existence_floor)
                else:
                    lh = likelihood_matrix[i, j]
                    if residual is not None:
                        cx, cy = det["centroid"]
                        clutter = self._clutter_intensity_local(residual, cx, cy)
                    else:
                        clutter = self._clutter_intensity()
                    r_new = (self.detection_prob * comp.r * lh) / (
                        self.detection_prob * comp.r * lh + clutter * (1 - comp.r) + 1e-10
                    )
                    if r_new < self.min_association_r:
                        # Association too weak — treat as missed
                        mc = deepcopy(comp)
                        if is_coasting:
                            mc.r = max(self.coast_existence_floor, comp.r * 0.97)
                        else:
                            mc.r = (1 - self.detection_prob) * comp.r
                        mc.consecutive_detections = 0
                        updated.append(mc)
                        used.discard(j)
                        continue
                cc = deepcopy(comp)
                cc.update(det["centroid"], det["intensity"])
                cc.r = r_new
                updated.append(cc)
            else:
                mc = deepcopy(comp)
                if is_coasting:
                    mc.r = max(self.coast_existence_floor, comp.r * 0.97)
                else:
                    mc.r = (1 - self.detection_prob) * comp.r
                mc.consecutive_detections = 0
                updated.append(mc)

        for j, det in enumerate(detections):
            if j not in used:
                pos = det["centroid"]
                state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
                P = np.eye(4, dtype=np.float64) * 100.0
                birth_r = 0.45 if det["intensity"] >= 50 else 0.15
                nc = BernoulliComponent(birth_r, state, P, track_id=None,
                                        process_noise_q=self.process_noise_q)
                nc.update(pos, det["intensity"])
                updated.append(nc)

        self.bernoulli_components = updated

    # ------------------------------------------------------------------
    def _clutter_intensity_local(self, residual: np.ndarray, cx: int, cy: int) -> float:
        base = self._clutter_intensity()
        r = 15
        h, w = residual.shape[:2]
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        patch = residual[y0:y1, x0:x1].astype(np.float32)
        local_std = float(np.std(patch)) if patch.size > 0 else 1.0
        noise_scale = 1.0 + min(3.0, local_std / 10.0)
        return base * noise_scale

    # ------------------------------------------------------------------
    # Pruning: coasting tracks live longer
    # ------------------------------------------------------------------
    def prune_and_merge(self):
        pruned = []
        for c in self.bernoulli_components:
            if c.track_id is not None and c.missed_detections <= self.coast_frames:
                if c.r > self.pruning_threshold * 0.3:
                    pruned.append(c)
            elif c.r > self.pruning_threshold:
                pruned.append(c)
        self.bernoulli_components = pruned

        for comp in self.bernoulli_components:
            promo_r = self.existence_threshold * 0.3 if comp.detection_count >= self.min_track_length + 2 else self.existence_threshold
            if (comp.r > promo_r
                    and comp.track_id is None
                    and comp.detection_count >= self.min_track_length):
                disp = comp.max_displacement
                avg_speed = disp / max(1, comp.age)
                min_disp = max(self.min_speed * 3, 12.0)
                linearity = _trajectory_linearity(comp)
                if avg_speed >= self.min_speed * 0.5 and disp >= min_disp and linearity >= 0.5:
                    comp.track_id = self.next_track_id
                    self.next_track_id += 1

        # Kill slow / non-physical components
        surviving = []
        for comp in self.bernoulli_components:
            if comp.age > 10 and comp.track_id is None:
                avg_spd = comp.max_displacement / comp.age
                if avg_spd < self.min_speed * 0.3:
                    continue
            if comp.track_id is not None and comp.age > 10:
                avg_spd = comp.max_displacement / comp.age
                linearity = _trajectory_linearity(comp)
                if avg_spd < self.min_speed * 0.4:
                    comp.track_id = None
                elif linearity < 0.3:
                    comp.track_id = None
            surviving.append(comp)
        self.bernoulli_components = surviving

        max_coast_age = 50 + self.coast_frames
        self.bernoulli_components = [
            c for c in self.bernoulli_components
            if not (c.age > max_coast_age and c.r < 0.3)
        ]

    # ------------------------------------------------------------------
    # Override: return RAW measurement centroid (not Kalman-smoothed)
    # ------------------------------------------------------------------
    def get_confirmed_tracks(self):
        tracks = []
        for comp in self.bernoulli_components:
            if (comp.r > self.existence_threshold
                    and comp.track_id is not None
                    and comp.detection_count >= self.min_track_length):
                speed = comp.get_speed()
                avg_speed = comp.max_displacement / max(1, comp.age) if comp.age > 3 else speed
                confidence = comp.calculate_confidence(
                    self.min_confidence_frames,
                    self.max_confidence_frames,
                )
                if not (self.min_speed <= speed <= self.max_speed):
                    continue
                if comp.age > 5 and avg_speed < self.min_speed * 0.7:
                    continue
                if comp.age > 8 and _trajectory_linearity(comp) < 0.35:
                    continue
                is_coasting = (comp.missed_detections > 0
                               and comp.missed_detections <= self.coast_frames)
                if is_coasting or confidence >= self.min_display_confidence:
                    if comp.missed_detections == 0:
                        raw = getattr(comp, "last_raw_measurement", None)
                        if raw is not None:
                            centroid = (int(round(raw[0])), int(round(raw[1])))
                        else:
                            centroid = comp.get_position()
                    else:
                        centroid = comp.get_position()
                    tracks.append({
                        "id": comp.track_id,
                        "centroid": centroid,
                        "existence_prob": comp.r,
                        "confidence": max(confidence, 0.55) if is_coasting else confidence,
                        "speed": speed,
                        "age": comp.age,
                        "detections": comp.detection_count,
                    })
        return tracks

    # ------------------------------------------------------------------
    # Override color mapping: range adjusted for nadir confidence values
    # ------------------------------------------------------------------
    def confidence_to_color(self, confidence):
        """Red (low) -> Yellow (mid) -> Green (high) in BGR."""
        lo, hi = 0.45, 0.85
        frac = max(0.0, min(1.0, (confidence - lo) / max(1e-9, hi - lo)))
        if frac < 0.5:
            t = frac * 2.0
            return (0, int(255 * t), 255)          # red → yellow
        t = (frac - 0.5) * 2.0
        return (0, 255, int(255 * (1.0 - t)))       # yellow → green

    # ------------------------------------------------------------------
    # Override annotation: bounding box + track ID label
    # ------------------------------------------------------------------
    def annotate_frame(self, frame, tracks):
        annotated = frame.copy()
        if annotated.ndim == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        box_half = 12  # half-size of bounding box in pixels

        for track in tracks:
            track_id = track["id"]
            centroid = track["centroid"]
            confidence = track["confidence"]
            color = self.confidence_to_color(confidence)

            self.track_history[track_id].append(centroid)
            self.track_colors[track_id].append(color)

            # Trail
            if len(self.track_history[track_id]) > 1:
                pts = list(self.track_history[track_id])
                cols = list(self.track_colors[track_id])
                for i in range(len(pts) - 1):
                    cv2.line(annotated, pts[i], pts[i + 1], cols[i + 1], 1)

            # Bounding box
            x, y = centroid
            tl = (x - box_half, y - box_half)
            br = (x + box_half, y + box_half)
            cv2.rectangle(annotated, tl, br, color, 2)

            # Centroid dot
            cv2.circle(annotated, centroid, 2, color, -1)

            # Label: "T<id>"
            label = f"T{track_id}"
            label_pos = (x - box_half, y - box_half - 5)
            cv2.putText(annotated, label, label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                        cv2.LINE_AA)

        return annotated

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------
    def run(self):
        if self.verbose:
            print("=" * 80)
            print("  NADIR PMB TRACKER  (space-looking-down, ego-motion compensated)")
            print("=" * 80)
            print(f"\n  Input:  {self.input_video_path}")
            if self.write_video_output:
                print(f"  Output: {self.output_video_path}")
            else:
                print("  Output video: disabled")

        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.input_video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.verbose:
            print(f"  {total_frames} frames  {width}x{height} @ {fps}fps")
            print(f"  Registration: {self.registration_method}  "
                  f"SNR_thresh={self.snr_threshold}")

        self.background = np.zeros((height, width), dtype=np.uint8)

        start = self.start_frame
        end = self.end_frame if self.end_frame else total_frames
        end = min(end, total_frames)

        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        out = None
        if self.write_video_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width * 2, height))

        if self.verbose:
            print(f"\n  PARAMETERS:")
            print(f"   Registration method:     {self.registration_method}")
            print(f"   SNR threshold:           {self.snr_threshold}")
            print(f"   Bright abs threshold:    {self.bright_abs_threshold}")
            print(f"   Peak min distance:       {self.peak_min_distance}")
            print(f"   Coast frames:            {self.coast_frames}")
            print(f"   Coast existence floor:   {self.coast_existence_floor}")
            print(f"   Existence threshold:     {self.existence_threshold}")
            print(f"   Min display confidence:  {self.min_display_confidence}")
            print(f"   Min track length:        {self.min_track_length}")
            print(f"   Max acceleration:        {self.max_acceleration}")
            print(f"   Max direction change:    {self.max_direction_change} deg")
            print(f"   Min/Max speed:           {self.min_speed}/{self.max_speed}")
            print("=" * 80)

        total_detections = 0
        track_log = []
        video_encode_s = 0.0
        loop_t0 = time.perf_counter()
        frames_processed = 0

        if self.verbose:
            print("\n  Processing frames...")

        for frame_idx in range(start, end):
            ret, bgr_frame = cap.read()
            if not ret:
                break

            gray = bgr_frame[:, :, 0] if bgr_frame.ndim == 3 else bgr_frame
            self.frame_buffer.append(gray)

            binary = self.preprocess_frame(gray)
            detections = self.detect_objects(binary, gray)
            total_detections += len(detections)

            self._frame_idx = frame_idx
            self.predict_components()
            self.update_components(detections)
            self.prune_and_merge()

            tracks = self.get_confirmed_tracks()

            if self.save_track_log_enabled:
                for t in tracks:
                    track_log.append({
                        "frame": frame_idx,
                        "track_id": t["id"],
                        "x": t["centroid"][0],
                        "y": t["centroid"][1],
                        "existence_prob": t["existence_prob"],
                        "confidence": t["confidence"],
                        "speed": t["speed"],
                        "age": t["age"],
                        "detections": t["detections"],
                    })

            if out is not None:
                ve0 = time.perf_counter()
                display_frame = bgr_frame.copy()
                if display_frame.ndim == 2:
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
                annotated = self.annotate_frame(display_frame, tracks)
                side_by_side = np.hstack([display_frame, annotated])
                out.write(side_by_side)
                video_encode_s += time.perf_counter() - ve0

            frames_processed += 1
            if self.verbose and (frames_processed % 50 == 0 or frame_idx == end - 1):
                nc = len(self.bernoulli_components)
                nt = len(tracks)
                reg = self._registration_stats[-1] if self._registration_stats else {}
                inliers_str = str(reg.get("inliers", "?"))
                print(f"   Frame {frames_processed}/{end - start}  "
                      f"dets={len(detections)}  comps={nc}  tracks={nt}  "
                      f"inliers={inliers_str}")

        cap.release()
        if out is not None:
            out.release()

        if self.save_track_log_enabled and track_log:
            self.save_track_log(track_log)

        loop_end = time.perf_counter()
        runtime_s = max(0.0, loop_end - loop_t0 - video_encode_s)
        unique_tracks = len(set(e["track_id"] for e in track_log)) if track_log else 0

        if self.verbose:
            print(f"\n  COMPLETE!")
            print(f"   Frames processed:  {frames_processed}")
            print(f"   Total detections:  {total_detections}")
            print(f"   Unique tracks:     {unique_tracks}")
            print(f"   Log entries:       {len(track_log)}")
            if self.write_video_output:
                print(f"   Output saved:      {self.output_video_path}")
                if video_encode_s > 0:
                    print(f"   Timing: tracking {runtime_s:.2f}s  "
                          f"video encode {video_encode_s:.2f}s (excluded)")
            print("=" * 80)

        return {
            "frames_processed": frames_processed,
            "total_detections": total_detections,
            "unique_tracks": unique_tracks,
            "log_entries": len(track_log),
            "runtime_s": round(runtime_s, 2),
            "video_encode_runtime_s": round(video_encode_s, 2),
            "registration_method": self.registration_method,
        }
