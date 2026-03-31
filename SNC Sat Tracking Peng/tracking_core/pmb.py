from __future__ import annotations

import time

import cv2
import numpy as np
from collections import defaultdict, deque
from scipy.ndimage import maximum_filter
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal, poisson
import scipy.stats as stats
from copy import deepcopy

from tracking_core.a2a_imu import (
    CameraIntrinsics,
    load_imu_samples_json,
    per_frame_imu_translation_lookup,
)
from tracking_core.a2a_preprocess import (
    apply_clahe_gray,
    stabilize_translation_phase_chain,
    warp_translate,
    warp_translate_color,
)


class BernoulliComponent:
    """
    Enhanced Bernoulli component with physical constraints.
    """
    def __init__(self, existence_prob, state, covariance, track_id=None):
        self.r = existence_prob  # Probability of existence
        self.state = state.copy()  # [x, y, vx, vy]
        self.P = covariance.copy()  # State covariance
        self.track_id = track_id
        self.history = deque(maxlen=50)
        self.age = 0
        self.detection_count = 0
        self.consecutive_detections = 0
        self.missed_detections = 0
        self.last_detection_frame = 0
        self.avg_intensity = 0
        self.max_displacement = 0
        self.first_position = state[:2].copy()
        
        # Kalman filter matrices
        dt = 1.0
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        
        q = 8.0  # Process noise
        self.Q = np.array([
            [q*dt**4/4, 0, q*dt**3/2, 0],
            [0, q*dt**4/4, 0, q*dt**3/2],
            [q*dt**3/2, 0, q*dt**2, 0],
            [0, q*dt**3/2, 0, q*dt**2]
        ], dtype=np.float64)
        
        self.R = np.eye(2, dtype=np.float64) * 2.0  # Measurement noise
    
    def predict(self):
        """Predict next state."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.missed_detections += 1
    
    def update(self, measurement, intensity=0):
        """Update with measurement."""
        z = np.array([measurement[0], measurement[1]], dtype=np.float64)
        
        # Innovation
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ self.H.T @ np.linalg.pinv(S)
        
        # Update state
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        self.detection_count += 1
        self.consecutive_detections += 1
        self.missed_detections = 0
        
        # Update average intensity
        if self.avg_intensity == 0:
            self.avg_intensity = intensity
        else:
            self.avg_intensity = 0.9 * self.avg_intensity + 0.1 * intensity
        
        # Update max displacement
        current_pos = self.state[:2]
        displacement = np.linalg.norm(current_pos - self.first_position)
        self.max_displacement = max(self.max_displacement, displacement)
        
        return self.get_position()
    
    def get_position(self):
        return (int(round(self.state[0])), int(round(self.state[1])))
    
    def get_velocity(self):
        return (self.state[2], self.state[3])
    
    def get_speed(self):
        return np.sqrt(self.state[2]**2 + self.state[3]**2)
    
    def calculate_confidence(self, min_confidence_frames=3, max_confidence_frames=15):
        """Calculate track confidence score."""
        # Base confidence from consecutive detections
        if self.consecutive_detections < min_confidence_frames:
            base = 0.2 + (self.consecutive_detections / min_confidence_frames) * 0.3
        else:
            effective = min(self.consecutive_detections, max_confidence_frames)
            base = 0.5 + 0.5 * (effective - min_confidence_frames) / \
                   (max_confidence_frames - min_confidence_frames)
        
        # Detection rate bonus
        if self.age > 0:
            detection_rate = self.detection_count / self.age
            base *= (0.7 + 0.3 * detection_rate)
        
        # Existence probability bonus
        base *= self.r
        
        # Intensity bonus
        if self.avg_intensity > 0:
            intensity_bonus = min(0.1, self.avg_intensity / 500)
            base += intensity_bonus
        
        return max(0.0, min(1.0, base))
    
    def check_physical_constraints(self, new_measurement, max_acceleration=30.0, 
                                   max_direction_change=70.0, max_speed=100.0):
        """Check if association violates physical constraints."""
        if self.detection_count < 2:
            return True
        
        new_pos = np.array([new_measurement[0], new_measurement[1]], dtype=np.float64)
        current_pos = self.state[:2]
        current_vel = self.state[2:]
        current_speed = self.get_speed()
        
        # Calculate implied velocity
        new_vel = new_pos - current_pos
        new_speed = np.linalg.norm(new_vel)
        
        # Check 1: Speed limit
        if new_speed > max_speed:
            return False
        
        # Check 2: Acceleration constraint (for established tracks)
        if self.consecutive_detections >= 3 and current_speed > 0.5:
            accel = np.linalg.norm(new_vel - current_vel)
            if accel > max_acceleration:
                return False
        
        # Check 3: Direction consistency (for fast-moving tracks)
        if self.consecutive_detections >= 3 and current_speed > 2.0 and new_speed > 2.0:
            cos_angle = np.dot(current_vel, new_vel) / (current_speed * new_speed + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_diff = np.arccos(cos_angle)
            
            if angle_diff > np.radians(max_direction_change):
                return False
        
        return True
    
    def likelihood(self, measurement):
        """Calculate measurement likelihood."""
        z = np.array([measurement[0], measurement[1]], dtype=np.float64)
        predicted_z = self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        
        try:
            return multivariate_normal.pdf(z, mean=predicted_z, cov=S)
        except:
            return 1e-10


class PoissonMultiBernoulliTracker:
    """
    Enhanced PMB filter with physical constraints and confidence filtering.
    """
    
    def __init__(
        self,
        input_video_path,
        output_video_path,
        
        # === PREPROCESSING ===
        bg_threshold=10,
        bg_sample_rate=50,
        use_temporal_smoothing=False,
        use_morphology=False,
        
        # === DETECTION ===
        cluster_eps=3,
        cluster_min_samples=1,
        min_detection_area=1,
        max_detection_area=300,
        min_intensity=25,
        
        # === PMB PARAMETERS ===
        birth_rate=0.1,
        survival_prob=0.99,
        detection_prob=0.85,
        clutter_rate=5.0,
        existence_threshold=0.5,
        pruning_threshold=0.01,
        
        # === PHYSICAL CONSTRAINTS ===
        max_acceleration=30.0,
        max_direction_change=70.0,
        max_speed=100.0,
        min_speed=0.3,
        
        # === QUALITY CONTROL ===
        min_track_length=4,
        min_confidence_frames=3,
        max_confidence_frames=15,
        min_display_confidence=0.60,
        
        # === DISPLAY ===
        track_history_length=10,
        
        # === FRAME RANGE ===
        start_frame=0,
        end_frame=None,
        
        # === OUTPUT ===
        save_track_log=True,
        track_log_path=None,
        verbose=True,
        write_video_output=True,

        # === AIR-TO-AIR (optional preprocessing before PMB) ===
        a2a_phase_stabilize=False,
        a2a_clahe_clip_limit=0.0,
        a2a_clahe_tile_size=8,
        a2a_imu_json_path=None,
        a2a_imu_apply_compensation=False,
        a2a_imu_fx=900.0,
        a2a_imu_fy=900.0,
        a2a_imu_default_dt=1.0 / 30.0,
    ):
        # Paths
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.verbose = verbose
        self.write_video_output = bool(write_video_output)
        
        # Preprocessing
        self.bg_threshold = bg_threshold
        self.bg_sample_rate = bg_sample_rate
        self.use_temporal_smoothing = use_temporal_smoothing
        self.use_morphology = use_morphology
        
        # Detection
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.min_detection_area = min_detection_area
        self.max_detection_area = max_detection_area
        self.min_intensity = min_intensity
        
        # PMB parameters
        self.birth_rate = birth_rate
        self.survival_prob = survival_prob
        self.detection_prob = detection_prob
        self.clutter_rate = clutter_rate
        self.existence_threshold = existence_threshold
        self.pruning_threshold = pruning_threshold
        
        # Physical constraints
        self.max_acceleration = max_acceleration
        self.max_direction_change = max_direction_change
        self.max_speed = max_speed
        self.min_speed = min_speed
        
        # Quality control
        self.min_track_length = min_track_length
        self.min_confidence_frames = min_confidence_frames
        self.max_confidence_frames = max_confidence_frames
        self.min_display_confidence = min_display_confidence
        
        # Display
        self.track_history_length = track_history_length
        
        # Frame range
        self.start_frame = start_frame
        self.end_frame = end_frame
        
        # Output
        self.save_track_log_enabled = save_track_log
        if track_log_path is None:
            self.track_log_path = output_video_path.replace('.mp4', '_pmb_tracks.txt')
        else:
            self.track_log_path = track_log_path
        
        # Air-to-air options
        self.a2a_phase_stabilize = bool(a2a_phase_stabilize)
        self.a2a_clahe_clip_limit = float(a2a_clahe_clip_limit)
        self.a2a_clahe_tile_size = int(a2a_clahe_tile_size)
        self.a2a_imu_json_path = a2a_imu_json_path
        self.a2a_imu_apply_compensation = bool(a2a_imu_apply_compensation)
        self.a2a_imu_fx = float(a2a_imu_fx)
        self.a2a_imu_fy = float(a2a_imu_fy)
        self.a2a_imu_default_dt = float(a2a_imu_default_dt)

        # Internal state
        self.bernoulli_components = []
        self.next_track_id = 1
        self.track_history = defaultdict(lambda: deque(maxlen=track_history_length))
        self.track_colors = defaultdict(lambda: deque(maxlen=track_history_length))
        self.background = None
        self.frame_buffer = deque(maxlen=3)
        self._a2a_display_shifts: list[tuple[float, float]] = []
        self.a2a_run_summary: dict = {}
        
    def compute_background(self, frames):
        """Compute background model."""
        if self.verbose:
            print(f"Computing background from {len(frames)} frames...")
        sample_frames = frames[::self.bg_sample_rate]
        stacked = np.stack(sample_frames, axis=0)
        background = np.median(stacked, axis=0)
        return background.astype(np.uint8)

    def _prepare_gray_sequence_air_to_air(self, all_frames: list) -> tuple[list, list[tuple[float, float]]]:
        """Optional CLAHE, IMU residual warp (stub model), and phase-correlation stabilization."""
        gray = [np.asarray(f, dtype=np.uint8).copy() for f in all_frames]
        n = len(gray)
        use_a2a = (
            self.a2a_phase_stabilize
            or self.a2a_clahe_clip_limit > 0
            or self.a2a_imu_apply_compensation
        )
        if not use_a2a:
            return gray, [(0.0, 0.0)] * n

        if self.a2a_clahe_clip_limit > 0:
            ts = (
                max(2, int(self.a2a_clahe_tile_size)),
                max(2, int(self.a2a_clahe_tile_size)),
            )
            gray = [apply_clahe_gray(g, self.a2a_clahe_clip_limit, ts) for g in gray]

        if self.a2a_imu_apply_compensation and self.a2a_imu_json_path:
            samples = load_imu_samples_json(self.a2a_imu_json_path)
            h, w = gray[0].shape[:2]
            intr = CameraIntrinsics(
                self.a2a_imu_fx,
                self.a2a_imu_fy,
                float(w) / 2.0,
                float(h) / 2.0,
            )
            tx, ty = per_frame_imu_translation_lookup(
                samples, len(gray), intr, self.a2a_imu_default_dt
            )
            gray = [warp_translate(gray[i], -tx[i], -ty[i]) for i in range(len(gray))]

        if self.a2a_phase_stabilize:
            gray, shifts = stabilize_translation_phase_chain(gray)
            return gray, shifts

        return gray, [(0.0, 0.0)] * len(gray)
    
    def preprocess_frame(self, frame):
        """Preprocess frame for detection."""
        if self.use_temporal_smoothing and len(self.frame_buffer) > 1:
            frame_smoothed = np.mean(list(self.frame_buffer), axis=0).astype(np.uint8)
        else:
            frame_smoothed = frame
        
        diff = cv2.absdiff(frame_smoothed, self.background)
        diff = cv2.GaussianBlur(diff, (3, 3), 0.3)
        _, binary = cv2.threshold(diff, self.bg_threshold, 255, cv2.THRESH_BINARY)
        
        if self.use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return binary
    
    def detect_objects(self, binary_frame, original_frame):
        """Detect objects with intensity filtering."""
        detections = []
        white_pixels = np.where(binary_frame == 255)
        
        if len(white_pixels[0]) == 0:
            return detections
        
        coords = list(zip(white_pixels[1], white_pixels[0]))
        
        if len(coords) == 0:
            return detections
        
        if len(coords) == 1:
            x, y = coords[0]
            intensity = float(original_frame[y, x])
            if intensity >= self.min_intensity:
                detections.append({
                    'centroid': (x, y),
                    'intensity': intensity
                })
        else:
            clustering = DBSCAN(eps=self.cluster_eps, 
                               min_samples=self.cluster_min_samples).fit(coords)
            
            for cluster_id in set(clustering.labels_):
                if cluster_id == -1:
                    continue
                    
                cluster_coords = [coords[i] for i in range(len(coords)) 
                                if clustering.labels_[i] == cluster_id]
                
                if len(cluster_coords) > 0:
                    area = len(cluster_coords)
                    
                    if area < self.min_detection_area or area > self.max_detection_area:
                        continue
                    
                    x_coords = [c[0] for c in cluster_coords]
                    y_coords = [c[1] for c in cluster_coords]
                    cx, cy = int(np.mean(x_coords)), int(np.mean(y_coords))
                    
                    intensities = [float(original_frame[y, x]) for x, y in cluster_coords]
                    max_intensity = np.max(intensities)
                    
                    if max_intensity >= self.min_intensity:
                        detections.append({
                            'centroid': (cx, cy),
                            'intensity': max_intensity
                        })
        
        return detections
    
    def predict_components(self):
        """Predict all Bernoulli components."""
        for component in self.bernoulli_components:
            component.predict()
            # Reduce existence probability due to possible death
            component.r = self.survival_prob * component.r
    
    def update_components(self, detections):
        """Update components with detections using data association and physical constraints."""
        n_components = len(self.bernoulli_components)
        n_detections = len(detections)
        
        if n_components == 0:
            # All detections are births
            for det in detections:
                pos = det['centroid']
                state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
                P = np.eye(4, dtype=np.float64) * 100.0
                component = BernoulliComponent(0.1, state, P, track_id=None)
                self.bernoulli_components.append(component)
            return
        
        if n_detections == 0:
            # Missed detection update for all components
            for component in self.bernoulli_components:
                component.r = (1 - self.detection_prob) * component.r
                component.consecutive_detections = 0
            return
        
        # Compute likelihood matrix with physical constraints
        likelihood_matrix = np.zeros((n_components, n_detections))
        valid_associations = np.ones((n_components, n_detections), dtype=bool)
        
        for i, component in enumerate(self.bernoulli_components):
            for j, det in enumerate(detections):
                # Check physical constraints
                if component.check_physical_constraints(
                    det['centroid'], 
                    self.max_acceleration,
                    self.max_direction_change,
                    self.max_speed
                ):
                    likelihood_matrix[i, j] = component.likelihood(det['centroid'])
                else:
                    valid_associations[i, j] = False
                    likelihood_matrix[i, j] = 1e-10
        
        # Data association
        clutter_intensity = self.clutter_rate / (128 * 128)
        
        updated_components = []
        used_detections = set()
        
        for i, component in enumerate(self.bernoulli_components):
            # Missed detection hypothesis
            r_miss = (1 - self.detection_prob) * component.r
            
            best_j = -1
            best_r = r_miss
            best_component = deepcopy(component)
            best_component.r = r_miss
            best_component.consecutive_detections = 0
            
            for j, det in enumerate(detections):
                if j in used_detections or not valid_associations[i, j]:
                    continue
                
                likelihood = likelihood_matrix[i, j]
                
                # Update existence probability
                r_update = (self.detection_prob * component.r * likelihood) / \
                          (self.detection_prob * component.r * likelihood + 
                           clutter_intensity * (1 - component.r) + 1e-10)
                
                if r_update > best_r and r_update > 0.1:
                    best_r = r_update
                    best_j = j
            
            if best_j >= 0:
                # Associated with detection
                component_copy = deepcopy(component)
                component_copy.update(detections[best_j]['centroid'], 
                                     detections[best_j]['intensity'])
                component_copy.r = best_r
                updated_components.append(component_copy)
                used_detections.add(best_j)
            else:
                # Missed detection
                updated_components.append(best_component)
        
        # Unassociated detections become births
        for j, det in enumerate(detections):
            if j not in used_detections:
                pos = det['centroid']
                state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
                P = np.eye(4, dtype=np.float64) * 100.0
                component = BernoulliComponent(0.15, state, P, track_id=None)
                component.update(pos, det['intensity'])
                updated_components.append(component)
        
        self.bernoulli_components = updated_components
    
    def prune_and_merge(self):
        """Prune low-probability components and assign track IDs."""
        # Prune
        self.bernoulli_components = [c for c in self.bernoulli_components 
                                     if c.r > self.pruning_threshold]
        
        # Assign track IDs to high-confidence components
        for component in self.bernoulli_components:
            if (component.r > self.existence_threshold and 
                component.track_id is None and
                component.detection_count >= self.min_track_length):
                component.track_id = self.next_track_id
                self.next_track_id += 1
        
        # Remove very old low-confidence components
        self.bernoulli_components = [c for c in self.bernoulli_components
                                     if not (c.age > 50 and c.r < 0.3)]
    
    def get_confirmed_tracks(self):
        """Extract confirmed tracks above thresholds."""
        tracks = []
        for component in self.bernoulli_components:
            if (component.r > self.existence_threshold and 
                component.track_id is not None and
                component.detection_count >= self.min_track_length):
                
                speed = component.get_speed()
                confidence = component.calculate_confidence(
                    self.min_confidence_frames,
                    self.max_confidence_frames
                )
                
                # Apply filters
                if (speed >= self.min_speed and 
                    speed <= self.max_speed and
                    confidence >= self.min_display_confidence):
                    
                    tracks.append({
                        'id': component.track_id,
                        'centroid': component.get_position(),
                        'existence_prob': component.r,
                        'confidence': confidence,
                        'speed': speed,
                        'age': component.age,
                        'detections': component.detection_count
                    })
        return tracks
    
    def confidence_to_color(self, confidence):
        """Convert confidence to BGR color."""
        confidence = max(0.0, min(1.0, confidence))
        if confidence < 0.5:
            t = confidence * 2
            return (0, int(255 * t), 255)  # Red -> Yellow
        else:
            t = (confidence - 0.5) * 2
            return (0, 255, int(255 * (1 - t)))  # Yellow -> Green
    
    def annotate_frame(self, frame, tracks):
        """Annotate frame with tracks."""
        annotated = frame.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
        
        for track in tracks:
            track_id = track['id']
            centroid = track['centroid']
            confidence = track['confidence']
            
            color = self.confidence_to_color(confidence)
            
            # Store history
            self.track_history[track_id].append(centroid)
            self.track_colors[track_id].append(color)
            
            # Draw trajectory
            if len(self.track_history[track_id]) > 1:
                points = list(self.track_history[track_id])
                colors = list(self.track_colors[track_id])
                for i in range(len(points) - 1):
                    cv2.line(annotated, points[i], points[i + 1], colors[i + 1], 1)
            
            # Draw centroid
            cv2.circle(annotated, centroid, 2, color, -1)
            
            # Draw bounding box
            cv2.rectangle(annotated, (centroid[0]-3, centroid[1]-3),
                         (centroid[0]+3, centroid[1]+3), color, 1)
        
        return annotated
    
    def save_track_log(self, track_log):
        """Save tracking log."""
        with open(self.track_log_path, 'w') as f:
            f.write("# Enhanced Poisson Multi-Bernoulli Tracking Log\n")
            f.write(f"# Input: {self.input_video_path}\n")
            f.write("#\n")
            f.write("# frame, track_id, x, y, existence_prob, confidence, speed, age, detections\n")
            f.write("#" + "=" * 80 + "\n")
            
            for entry in track_log:
                f.write(f"{entry['frame']}, {entry['track_id']}, "
                       f"{entry['x']}, {entry['y']}, "
                       f"{entry['existence_prob']:.4f}, {entry['confidence']:.4f}, "
                       f"{entry['speed']:.4f}, {entry['age']}, {entry['detections']}\n")
        
        if self.verbose:
            print(f"   📝 Track log saved: {self.track_log_path}")
    
    def run(self):
        """Execute the enhanced PMB tracking pipeline."""
        if self.verbose:
            print("=" * 80)
            print("🛰️  ENHANCED POISSON MULTI-BERNOULLI TRACKER")
            print("=" * 80)
            print(f"\n📹 Input:  {self.input_video_path}")
            if self.write_video_output:
                print(f"📹 Output: {self.output_video_path}")
            else:
                print("📹 Output video: disabled (tracking-only)")
        
        # Load video
        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.input_video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Load all frames
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = frame[:, :, 0] if len(frame.shape) == 3 else frame
            all_frames.append(gray)
        cap.release()
        
        if self.verbose:
            print(f"   Loaded {len(all_frames)} frames ({width}x{height} @ {fps}fps)")

        processed_all, self._a2a_display_shifts = self._prepare_gray_sequence_air_to_air(all_frames)
        if self.verbose and (
            self.a2a_phase_stabilize
            or self.a2a_clahe_clip_limit > 0
            or self.a2a_imu_apply_compensation
        ):
            print(
                "   A2A preprocess: "
                f"phase_stabilize={self.a2a_phase_stabilize}, "
                f"clahe_clip={self.a2a_clahe_clip_limit}, "
                f"imu_compensation={self.a2a_imu_apply_compensation}"
            )
        
        # Frame range
        start = self.start_frame
        end = self.end_frame if self.end_frame else len(all_frames)
        end = min(end, len(all_frames))
        frames_to_process = processed_all[start:end]
        
        # Compute background (on A2A-processed stack so median matches stabilized coordinates)
        self.background = self.compute_background(processed_all)
        
        if self.verbose:
            print(f"\n⚙️  PARAMETERS:")
            print(f"   Existence threshold:     {self.existence_threshold}")
            print(f"   Min display confidence:  {self.min_display_confidence}")
            print(f"   Min track length:        {self.min_track_length}")
            print(f"   Max acceleration:        {self.max_acceleration}")
            print(f"   Max direction change:    {self.max_direction_change}°")
            print(f"   Min/Max speed:           {self.min_speed}/{self.max_speed}")
            print("=" * 80)
        
        out = None
        if self.write_video_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width * 2, height))

        # Reopen for processing
        cap = cv2.VideoCapture(self.input_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        total_detections = 0
        track_log = []

        if self.verbose:
            print("\n🎬 Processing frames...")

        video_encode_s = 0.0
        loop_t0 = time.perf_counter()

        for i, gray_frame in enumerate(frames_to_process):
            ret, original_frame = cap.read()
            if not ret:
                break
            
            # Add to buffer
            self.frame_buffer.append(gray_frame)
            
            # Preprocess and detect
            binary = self.preprocess_frame(gray_frame)
            detections = self.detect_objects(binary, gray_frame)
            total_detections += len(detections)
            
            # PMB prediction
            self.predict_components()
            
            # PMB update with constraints
            self.update_components(detections)
            
            # Prune and merge
            self.prune_and_merge()
            
            # Get confirmed tracks
            tracks = self.get_confirmed_tracks()
            
            # Log tracks
            if self.save_track_log_enabled:
                for track in tracks:
                    track_log.append({
                        'frame': start + i,
                        'track_id': track['id'],
                        'x': track['centroid'][0],
                        'y': track['centroid'][1],
                        'existence_prob': track['existence_prob'],
                        'confidence': track['confidence'],
                        'speed': track['speed'],
                        'age': track['age'],
                        'detections': track['detections']
                    })
            
            if out is not None:
                ve0 = time.perf_counter()
                gidx = start + i
                tx_ty = self._a2a_display_shifts[gidx] if gidx < len(self._a2a_display_shifts) else (0.0, 0.0)
                display_frame = warp_translate_color(original_frame, tx_ty[0], tx_ty[1])
                annotated = self.annotate_frame(display_frame, tracks)
                side_by_side = np.hstack([display_frame, annotated])
                out.write(side_by_side)
                video_encode_s += time.perf_counter() - ve0
            
            # Progress
            if self.verbose and ((i + 1) % 100 == 0 or i == len(frames_to_process) - 1):
                n_components = len(self.bernoulli_components)
                n_confirmed = len(tracks)
                print(f"   Frame {i+1}/{len(frames_to_process)} - "
                      f"Components: {n_components}, "
                      f"Confirmed: {n_confirmed}, "
                      f"Detections: {len(detections)}")
        
        cap.release()
        if out is not None:
            out.release()

        if self.save_track_log_enabled and track_log:
            self.save_track_log(track_log)

        loop_end = time.perf_counter()
        runtime_s = max(0.0, loop_end - loop_t0 - video_encode_s)

        # Summary
        unique_tracks = len(set(e['track_id'] for e in track_log)) if track_log else 0

        if self.verbose:
            print(f"\n🎉 COMPLETE!")
            print(f"   Frames processed:  {len(frames_to_process)}")
            print(f"   Total detections:  {total_detections}")
            print(f"   Unique tracks:     {unique_tracks}")
            print(f"   Log entries:       {len(track_log)}")
            if self.write_video_output:
                print(f"   Output saved:      {self.output_video_path}")
                if video_encode_s > 0:
                    print(
                        f"   Timing:            tracking {runtime_s:.2f}s, "
                        f"video encode {video_encode_s:.2f}s (excluded from runtime_s)"
                    )
            print("=" * 80)

        self.a2a_run_summary = {
            "a2a_phase_stabilize": self.a2a_phase_stabilize,
            "a2a_clahe_clip_limit": self.a2a_clahe_clip_limit,
            "a2a_imu_apply_compensation": self.a2a_imu_apply_compensation,
            "a2a_imu_json_path": self.a2a_imu_json_path,
        }
        
        return {
            'frames_processed': len(frames_to_process),
            'total_detections': total_detections,
            'unique_tracks': unique_tracks,
            'log_entries': len(track_log),
            'air_to_air': self.a2a_run_summary,
            'runtime_s': round(runtime_s, 2),
            'video_encode_runtime_s': round(video_encode_s, 2),
        }


def _mahalanobis_sq_position(component: BernoulliComponent, centroid_xy: tuple[float, float] | tuple[int, int]) -> float:
    """Squared Mahalanobis distance of measurement to Gaussian predicted position (2D)."""
    z = np.array([float(centroid_xy[0]), float(centroid_xy[1])], dtype=np.float64)
    y = z - component.H @ component.state
    S = component.H @ component.P @ component.H.T + component.R
    try:
        return float(y @ np.linalg.solve(S, y))
    except np.linalg.LinAlgError:
        return float(y @ np.linalg.lstsq(S, y, rcond=None)[0])


class TextbookPoissonMultiBernoulliTracker(PoissonMultiBernoulliTracker):
    """
    Reference-style PMB path for side-by-side comparison with **Current PMB** (greedy + heuristics).

    Differences from :class:`PoissonMultiBernoulliTracker`:

    - **Gating:** innovation (Mahalanobis) gate on :math:`\\nu^\\top S^{-1} \\nu` instead of
      turn/accel/speed heuristics (``check_physical_constraints`` not used here).
    - **Clutter:** intensity :math:`\\lambda_c = \\texttt{clutter_rate} / (H \\cdot W)` when
      background geometry is known; otherwise falls back to ``clutter_rate / (128^2)``.
    - **Association:** one-to-one assignment via :func:`scipy.optimize.linear_sum_assignment`
      on a rectangular cost matrix with one **miss** column per track (global optimum for
      that linear cost model), instead of list-order greedy best-detection.
    - **Births:** initial Bernoulli existence from Poisson-style prior
      ``birth_rate / (birth_rate + λ_c)`` (capped), then standard Kalman update to the
      measurement — uses the tracker ``birth_rate`` hyperparameter in the recursion.
    - **Output:** display **confidence** is set to **existence** :math:`r` (no separate
      heuristic confidence curve), subject to ``min_display_confidence`` and speed bounds.
    """

    def __init__(
        self,
        input_video_path,
        output_video_path,
        textbook_mahalanobis_gate_sq: float = 9.21,
        **kwargs,
    ):
        # 9.21 ≈ chi-square 99% for 2 DOF (position-only measurement).
        self.textbook_mahalanobis_gate_sq = float(textbook_mahalanobis_gate_sq)
        super().__init__(input_video_path, output_video_path, **kwargs)

    def _lambda_c(self) -> float:
        if self.background is not None:
            h, w = self.background.shape[:2]
            return self.clutter_rate / float(max(1, h * w))
        return self.clutter_rate / (128.0 * 128.0)

    def _r_update_associated(self, component: BernoulliComponent, likelihood: float) -> float:
        pd = self.detection_prob
        r = float(component.r)
        lam = self._lambda_c()
        num = pd * r * likelihood
        den = num + lam * (1.0 - r) + 1e-16
        return float(np.clip(num / den, 0.0, 1.0))

    def _poisson_birth_existence(self) -> float:
        lam = self._lambda_c()
        br = max(float(self.birth_rate), 1e-9)
        r0 = br / (br + lam + 1e-12)
        return float(np.clip(r0, 0.02, 0.55))

    def _textbook_miss_association_cost(self) -> float:
        """Hungarian cost for 'track i not associated'; must exceed typical -log L for real targets."""
        return 18.0

    def update_components(self, detections):
        """Mahalanobis gate + Hungarian assignment + Bernoulli miss/update + Poisson-style births."""
        n_components = len(self.bernoulli_components)
        n_detections = len(detections)
        lambda_c = self._lambda_c()

        if n_components == 0:
            r0 = self._poisson_birth_existence()
            for det in detections:
                pos = det["centroid"]
                state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
                P = np.eye(4, dtype=np.float64) * 100.0
                born = BernoulliComponent(r0, state, P, track_id=None)
                born.update(pos, det.get("intensity", 0.0))
                self.bernoulli_components.append(born)
            return

        if n_detections == 0:
            for component in self.bernoulli_components:
                component.r = (1.0 - self.detection_prob) * component.r
                component.consecutive_detections = 0
            return

        BIG = 1.0e9
        M = n_components
        N = n_detections
        ncol = N + M
        C = np.full((M, ncol), BIG, dtype=np.float64)

        likelihoods = np.zeros((M, N), dtype=np.float64)
        gated = np.zeros((M, N), dtype=bool)

        for i, component in enumerate(self.bernoulli_components):
            for j, det in enumerate(detections):
                m2 = _mahalanobis_sq_position(component, det["centroid"])
                if m2 <= self.textbook_mahalanobis_gate_sq:
                    L = float(component.likelihood(det["centroid"]))
                    likelihoods[i, j] = max(L, 1e-30)
                    gated[i, j] = True
                    C[i, j] = -np.log(likelihoods[i, j])
            C[i, N + i] = self._textbook_miss_association_cost()

        row_ind, col_ind = linear_sum_assignment(C)

        col_for_row: dict[int, int] = {}
        for ri, ci in zip(row_ind.tolist(), col_ind.tolist()):
            col_for_row[int(ri)] = int(ci)

        used_det_cols: set[int] = set()
        updated_components: list[BernoulliComponent] = []

        for i, component in enumerate(self.bernoulli_components):
            ccol = col_for_row.get(i, N + i)
            if ccol >= N:
                missed = _clone_bernoulli_like(component)
                missed.r = (1.0 - self.detection_prob) * component.r
                missed.consecutive_detections = 0
                updated_components.append(missed)
                continue

            j = ccol
            if not gated[i, j] or C[i, j] >= BIG * 0.5:
                missed = _clone_bernoulli_like(component)
                missed.r = (1.0 - self.detection_prob) * component.r
                missed.consecutive_detections = 0
                updated_components.append(missed)
                continue

            L = likelihoods[i, j]
            r_new = self._r_update_associated(component, L)
            comp_copy = deepcopy(component)
            comp_copy.update(detections[j]["centroid"], detections[j].get("intensity", 0.0))
            comp_copy.r = r_new
            updated_components.append(comp_copy)
            used_det_cols.add(j)

        r0 = self._poisson_birth_existence()
        for j, det in enumerate(detections):
            if j in used_det_cols:
                continue
            pos = det["centroid"]
            state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
            P = np.eye(4, dtype=np.float64) * 100.0
            born = BernoulliComponent(r0, state, P, track_id=None)
            born.update(pos, det.get("intensity", 0.0))
            updated_components.append(born)

        self.bernoulli_components = updated_components

    def get_confirmed_tracks(self):
        """Confirmed tracks: use existence *r* as display confidence (textbook-style)."""
        tracks = []
        for component in self.bernoulli_components:
            if (
                component.r > self.existence_threshold
                and component.track_id is not None
                and component.detection_count >= self.min_track_length
            ):
                speed = component.get_speed()
                conf = float(component.r)
                if (
                    speed >= self.min_speed
                    and speed <= self.max_speed
                    and conf >= self.min_display_confidence
                ):
                    tracks.append(
                        {
                            "id": component.track_id,
                            "centroid": component.get_position(),
                            "existence_prob": component.r,
                            "confidence": conf,
                            "speed": speed,
                            "age": component.age,
                            "detections": component.detection_count,
                        }
                    )
        return tracks

    def save_track_log(self, track_log):
        with open(self.track_log_path, "w", encoding="utf-8") as f:
            f.write("# Textbook Poisson Multi-Bernoulli Tracking Log (Hungarian + Mahalanobis gate)\n")
            f.write(f"# Input: {self.input_video_path}\n")
            f.write("#\n")
            f.write("# frame, track_id, x, y, existence_prob, confidence, speed, age, detections\n")
            f.write("#" + "=" * 80 + "\n")
            for entry in track_log:
                f.write(
                    f"{entry['frame']}, {entry['track_id']}, "
                    f"{entry['x']}, {entry['y']}, "
                    f"{entry['existence_prob']:.4f}, {entry['confidence']:.4f}, "
                    f"{entry['speed']:.4f}, {entry['age']}, {entry['detections']}\n"
                )
        if self.verbose:
            print(f"   📝 Track log saved: {self.track_log_path}")


def _clone_bernoulli_like(src: BernoulliComponent, r_override: float | None = None) -> BernoulliComponent:
    """Lightweight Bernoulli copy (avoids deepcopy on the hot path)."""
    r = float(src.r if r_override is None else r_override)
    c = BernoulliComponent(r, src.state, src.P, src.track_id)
    c.age = src.age
    c.detection_count = src.detection_count
    c.consecutive_detections = src.consecutive_detections
    c.missed_detections = src.missed_detections
    c.last_detection_frame = src.last_detection_frame
    c.avg_intensity = src.avg_intensity
    c.max_displacement = src.max_displacement
    c.first_position = src.first_position.copy()
    maxlen = src.history.maxlen if src.history.maxlen is not None else 50
    c.history = deque(list(src.history), maxlen=maxlen)
    return c


class SparseBudgetPmTracker(PoissonMultiBernoulliTracker):
    """
    Throughput-oriented PMB-family tracker: same greedy Bernoulli semantics as Current PMB,
    but with (optional) streaming I/O, pixel-gated sparse likelihood evaluation, caps on
    live components and detections per frame, frame-scaled clutter intensity, and stricter
    birth filtering for weak detections.

    When any air-to-air preprocessing flag is enabled, falls back to the standard
    ``PoissonMultiBernoulliTracker.run()`` path (full-frame load + A2A stack).
    """

    def __init__(self, **kwargs):
        self.pmb_max_detections_per_frame = max(1, int(kwargs.pop("pmb_max_detections_per_frame", 96)))
        self.pmb_max_live_components = max(1, int(kwargs.pop("pmb_max_live_components", 140)))
        self.pmb_assoc_gate_pixels = float(kwargs.pop("pmb_assoc_gate_pixels", 48.0))
        self.pmb_min_birth_intensity = float(kwargs.pop("pmb_min_birth_intensity", 34.0))
        self.pmb_streaming_run = bool(kwargs.pop("pmb_streaming_run", True))
        self.pmb_write_video_output = bool(kwargs.pop("pmb_write_video_output", True))
        self.pmb_frame_scaled_clutter = bool(kwargs.pop("pmb_frame_scaled_clutter", True))
        super().__init__(**kwargs)

    def _clutter_intensity(self) -> float:
        if not self.pmb_frame_scaled_clutter or self.background is None:
            return self.clutter_rate / (128 * 128)
        h, w = self.background.shape[:2]
        return self.clutter_rate / float(max(1, h * w))

    def _cap_detections(self, detections: list) -> list:
        if len(detections) <= self.pmb_max_detections_per_frame:
            return detections
        sorted_d = sorted(detections, key=lambda d: d.get("intensity", 0.0), reverse=True)
        return sorted_d[: self.pmb_max_detections_per_frame]

    def _cap_live_components(self) -> None:
        if len(self.bernoulli_components) <= self.pmb_max_live_components:
            return
        ordered = sorted(self.bernoulli_components, key=lambda c: c.r, reverse=True)
        self.bernoulli_components = ordered[: self.pmb_max_live_components]

    def update_components(self, detections):
        """Greedy PMB update with spatial gating and budgets (no dense likelihood matrix)."""
        detections = self._cap_detections(detections)
        self._cap_live_components()

        n_components = len(self.bernoulli_components)
        n_detections = len(detections)

        if n_components == 0:
            for det in detections:
                if det.get("intensity", 0.0) < self.pmb_min_birth_intensity:
                    continue
                pos = det["centroid"]
                state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
                P = np.eye(4, dtype=np.float64) * 100.0
                component = BernoulliComponent(0.1, state, P, track_id=None)
                self.bernoulli_components.append(component)
            return

        if n_detections == 0:
            for component in self.bernoulli_components:
                component.r = (1 - self.detection_prob) * component.r
                component.consecutive_detections = 0
            return

        clutter_intensity = self._clutter_intensity()
        gate2 = self.pmb_assoc_gate_pixels ** 2
        H = self.bernoulli_components[0].H

        updated_components: list = []
        used_detections: set[int] = set()

        # Greedy one-to-one: process higher-existence components first so real tracks
        # claim detections before weak ghosts (cost-aware order vs. raw list order).
        comp_order = sorted(
            range(n_components),
            key=lambda idx: self.bernoulli_components[idx].r,
            reverse=True,
        )

        for i in comp_order:
            component = self.bernoulli_components[i]
            r_miss = (1 - self.detection_prob) * component.r
            best_j = -1
            best_r = r_miss
            best_component = _clone_bernoulli_like(component, r_override=r_miss)
            best_component.consecutive_detections = 0

            pred = H @ component.state
            zx, zy = float(pred[0]), float(pred[1])

            for j, det in enumerate(detections):
                cx, cy = det["centroid"]
                if (cx - zx) ** 2 + (cy - zy) ** 2 > gate2:
                    continue
                if not component.check_physical_constraints(
                    det["centroid"],
                    self.max_acceleration,
                    self.max_direction_change,
                    self.max_speed,
                ):
                    continue
                lik = component.likelihood(det["centroid"])
                r_update = (self.detection_prob * component.r * lik) / (
                    self.detection_prob * component.r * lik
                    + clutter_intensity * (1 - component.r)
                    + 1e-10
                )
                if r_update > best_r and r_update > 0.1:
                    best_r = r_update
                    best_j = j

            if best_j >= 0:
                component_copy = _clone_bernoulli_like(component)
                component_copy.update(detections[best_j]["centroid"], detections[best_j]["intensity"])
                component_copy.r = best_r
                updated_components.append(component_copy)
                used_detections.add(best_j)
            else:
                updated_components.append(best_component)

        for j, det in enumerate(detections):
            if j in used_detections:
                continue
            if det.get("intensity", 0.0) < self.pmb_min_birth_intensity:
                continue
            pos = det["centroid"]
            state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
            P = np.eye(4, dtype=np.float64) * 100.0
            born = BernoulliComponent(0.15, state, P, track_id=None)
            born.update(pos, det["intensity"])
            updated_components.append(born)

        self.bernoulli_components = updated_components

    def prune_and_merge(self):
        """Slightly more aggressive pruning for unconfirmed low-r ghosts before base logic."""
        kept: list = []
        for c in self.bernoulli_components:
            if c.track_id is None and c.r < 0.06 and c.age > 18 and c.consecutive_detections == 0:
                continue
            kept.append(c)
        self.bernoulli_components = kept
        super().prune_and_merge()

    def run(self):
        use_a2a = (
            self.a2a_phase_stabilize
            or self.a2a_clahe_clip_limit > 0
            or self.a2a_imu_apply_compensation
        )
        if use_a2a or not self.pmb_streaming_run:
            if self.verbose and use_a2a and self.pmb_streaming_run:
                print(
                    "   Sparse PMB: A2A preprocessing enabled — using full-memory PMB run() "
                    "(streaming path requires A2A off)."
                )
            return super().run()
        return self._run_streaming()

    def _run_streaming(self):
        """Two-pass streaming: sample background without storing all frames, then track."""
        if self.verbose:
            print("=" * 80)
            print("🛰️  SPARSE BUDGET PMB (streaming)")
            print("=" * 80)
            print(f"\n📹 Input:  {self.input_video_path}")
            if self.write_video_output and self.pmb_write_video_output:
                print(f"📹 Output: {self.output_video_path}")
            else:
                print("📹 Output video: disabled (tracking-only or pmb_write_video_output=false)")

        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.input_video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        samples: list[np.ndarray] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.bg_sample_rate == 0:
                gray = frame[:, :, 0] if len(frame.shape) == 3 else frame
                samples.append(np.asarray(gray, dtype=np.uint8))
            idx += 1
        n_total = idx
        cap.release()

        if not samples:
            raise ValueError("No frames read for background model.")

        stacked = np.stack(samples, axis=0)
        self.background = np.median(stacked, axis=0).astype(np.uint8)

        start = self.start_frame
        end = self.end_frame if self.end_frame is not None else n_total
        end = min(end, n_total)
        if start >= end:
            raise ValueError(f"Invalid frame range: start={start}, end={end}")

        self._a2a_display_shifts = [(0.0, 0.0)] * max(n_total, end)

        if self.verbose:
            print(f"   Frames in file: {n_total} ({width}x{height} @ {fps}fps)")
            print(f"   Background from {len(samples)} samples (every {self.bg_sample_rate} frames)")
            print(
                f"   Sparse PMB budgets: max_dets={self.pmb_max_detections_per_frame}, "
                f"max_components={self.pmb_max_live_components}, gate_px={self.pmb_assoc_gate_pixels}"
            )
            print(f"\n⚙️  PARAMETERS:")
            print(f"   Existence threshold:     {self.existence_threshold}")
            print(f"   Min display confidence:  {self.min_display_confidence}")
            print(f"   Min track length:        {self.min_track_length}")
            print("=" * 80)

        out = None
        if self.write_video_output and self.pmb_write_video_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width * 2, height))

        cap = cv2.VideoCapture(self.input_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        total_detections = 0
        track_log: list = []
        n_proc = end - start

        if self.verbose:
            print("\n🎬 Processing frames (streaming)...")

        video_encode_s = 0.0
        loop_t0 = time.perf_counter()

        for i in range(n_proc):
            ret, original_frame = cap.read()
            if not ret:
                break
            gray_frame = (
                original_frame[:, :, 0]
                if len(original_frame.shape) == 3
                else original_frame
            )
            gray_frame = np.asarray(gray_frame, dtype=np.uint8)

            self.frame_buffer.append(gray_frame)
            binary = self.preprocess_frame(gray_frame)
            detections = self.detect_objects(binary, gray_frame)
            total_detections += len(detections)

            self.predict_components()
            self.update_components(detections)
            self.prune_and_merge()
            tracks = self.get_confirmed_tracks()

            frame_idx = start + i
            if self.save_track_log_enabled:
                for track in tracks:
                    track_log.append(
                        {
                            "frame": frame_idx,
                            "track_id": track["id"],
                            "x": track["centroid"][0],
                            "y": track["centroid"][1],
                            "existence_prob": track["existence_prob"],
                            "confidence": track["confidence"],
                            "speed": track["speed"],
                            "age": track["age"],
                            "detections": track["detections"],
                        }
                    )

            if self.write_video_output and self.pmb_write_video_output and out is not None:
                ve0 = time.perf_counter()
                gidx = frame_idx
                tx_ty = (
                    self._a2a_display_shifts[gidx]
                    if gidx < len(self._a2a_display_shifts)
                    else (0.0, 0.0)
                )
                display_frame = warp_translate_color(original_frame, tx_ty[0], tx_ty[1])
                annotated = self.annotate_frame(display_frame, tracks)
                side_by_side = np.hstack([display_frame, annotated])
                out.write(side_by_side)
                video_encode_s += time.perf_counter() - ve0

            if self.verbose and ((i + 1) % 100 == 0 or i == n_proc - 1):
                print(
                    f"   Frame {i + 1}/{n_proc} - "
                    f"Components: {len(self.bernoulli_components)}, "
                    f"Confirmed: {len(tracks)}, "
                    f"Detections(raw): {len(detections)}"
                )

        cap.release()
        if out is not None:
            out.release()

        if self.save_track_log_enabled and track_log:
            self.save_track_log(track_log)

        loop_end = time.perf_counter()
        runtime_s = max(0.0, loop_end - loop_t0 - video_encode_s)

        unique_tracks = len({e["track_id"] for e in track_log}) if track_log else 0

        if self.verbose:
            print(f"\n🎉 COMPLETE (sparse streaming)")
            print(f"   Frames processed:  {n_proc}")
            print(f"   Total detections:  {total_detections}")
            print(f"   Unique tracks:     {unique_tracks}")
            print(f"   Log entries:       {len(track_log)}")
            if self.write_video_output and self.pmb_write_video_output:
                print(f"   Output saved:      {self.output_video_path}")
                if video_encode_s > 0:
                    print(
                        f"   Timing:            tracking {runtime_s:.2f}s, "
                        f"video encode {video_encode_s:.2f}s (excluded from runtime_s)"
                    )
            print("=" * 80)

        self.a2a_run_summary = {
            "a2a_phase_stabilize": self.a2a_phase_stabilize,
            "a2a_clahe_clip_limit": self.a2a_clahe_clip_limit,
            "a2a_imu_apply_compensation": self.a2a_imu_apply_compensation,
            "a2a_imu_json_path": self.a2a_imu_json_path,
            "pmb_streaming": True,
            "pmb_sparse_update": True,
        }

        return {
            "frames_processed": n_proc,
            "total_detections": total_detections,
            "unique_tracks": unique_tracks,
            "log_entries": len(track_log),
            "air_to_air": self.a2a_run_summary,
            "runtime_s": round(runtime_s, 2),
            "video_encode_runtime_s": round(video_encode_s, 2),
        }


class TrueTbdPmTracker(PoissonMultiBernoulliTracker):
    """
    Multi-target track-before-detect style tracker on passive optical residuals.

    Builds a per-frame soft fused residual map (temporal max/mean, no global
    binary mask). Birth proposals come from local maxima with an adaptive
    (per-frame percentile) floor. PMB data association uses Gaussian centroid
    likelihood scaled by local integrated evidence — no hard thresholded
    detection image in the loop.
    """

    def __init__(
        self,
        input_video_path,
        output_video_path,
        tbd_window=4,
        tbd_evidence_percentile=87.0,
        tbd_evidence_floor=1.5,
        tbd_peak_min_distance=5,
        tbd_soft_likelihood_gain=1.25,
        tbd_local_patch_radius=2,
        tbd_max_proposals_per_frame=24,
        **kwargs,
    ):
        super().__init__(input_video_path=input_video_path, output_video_path=output_video_path, **kwargs)
        self.tbd_window = max(1, int(tbd_window))
        self.tbd_evidence_percentile = float(tbd_evidence_percentile)
        self.tbd_evidence_floor = float(tbd_evidence_floor)
        self.tbd_peak_min_distance = max(1, int(tbd_peak_min_distance))
        self.tbd_soft_likelihood_gain = float(tbd_soft_likelihood_gain)
        self.tbd_local_patch_radius = max(1, int(tbd_local_patch_radius))
        self.tbd_max_proposals_per_frame = max(1, int(tbd_max_proposals_per_frame))
        self.tbd_evidence_buffer: deque = deque(maxlen=self.tbd_window)
        self._last_evidence: np.ndarray | None = None
        # Per-frame caches (avoid repeated full-frame percentiles in association loop)
        self._tbd_cached_ref: float = 1.0
        self._tbd_cached_peak_gate: float = 0.0
        self._tbd_cached_foot: float = 0.0

    def preprocess_frame(self, frame):
        if self.use_temporal_smoothing and len(self.frame_buffer) > 1:
            frame_smoothed = np.mean(list(self.frame_buffer), axis=0).astype(np.uint8)
        else:
            frame_smoothed = frame

        diff = cv2.absdiff(frame_smoothed, self.background).astype(np.float32)
        diff = cv2.GaussianBlur(diff, (3, 3), 0.3)
        self.tbd_evidence_buffer.append(diff)

        if len(self.tbd_evidence_buffer) > 1:
            stacked = np.stack(self.tbd_evidence_buffer, axis=0)
            fused = 0.6 * np.max(stacked, axis=0) + 0.4 * np.mean(stacked, axis=0)
        else:
            fused = diff

        self._last_evidence = fused.astype(np.float32, copy=False)
        e = self._last_evidence
        patch_area = (2 * self.tbd_local_patch_radius + 1) ** 2
        p90 = float(np.percentile(e, 90))
        self._tbd_cached_ref = max(self.tbd_evidence_floor, p90) * patch_area + 1e-6
        pp = max(50.0, self.tbd_evidence_percentile - 20.0)
        self._tbd_cached_peak_gate = max(
            self.tbd_evidence_floor,
            float(np.percentile(e, pp)),
        )
        self._tbd_cached_foot = max(
            self.tbd_evidence_floor,
            float(np.percentile(e, self.tbd_evidence_percentile)),
        )
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    def _local_evidence_sum(self, evidence: np.ndarray, cx: float, cy: float) -> float:
        r = self.tbd_local_patch_radius
        xi, yi = int(round(cx)), int(round(cy))
        h, w = evidence.shape
        x0, x1 = max(0, xi - r), min(w, xi + r + 1)
        y0, y1 = max(0, yi - r), min(h, yi + r + 1)
        if x0 >= x1 or y0 >= y1:
            return 0.0
        return float(np.sum(evidence[y0:y1, x0:x1]))

    def _evidence_reference_level(self, evidence: np.ndarray) -> float:
        if evidence is self._last_evidence:
            return self._tbd_cached_ref
        patch_area = (2 * self.tbd_local_patch_radius + 1) ** 2
        baseline = max(
            self.tbd_evidence_floor,
            float(np.percentile(evidence, 90)),
        )
        return baseline * patch_area + 1e-6

    def _evidence_likelihood_scale(self, component: BernoulliComponent, det_centroid) -> float:
        if self._last_evidence is None:
            return 1.0
        e = self._last_evidence
        predicted_x, predicted_y = component.state[:2]
        predicted_sum = self._local_evidence_sum(e, predicted_x, predicted_y)
        detection_sum = self._local_evidence_sum(e, det_centroid[0], det_centroid[1])
        ref = self._evidence_reference_level(e)
        predicted_ratio = predicted_sum / ref
        detection_ratio = detection_sum / ref
        combined_ratio = 0.7 * predicted_ratio + 0.3 * detection_ratio
        boost = 1.0 + self.tbd_soft_likelihood_gain * min(4.0, combined_ratio)
        return float(np.clip(boost, 0.2, 12.0))

    def _passes_birth_gate(self, raw_intensity: float, evidence_sum: float, peak_value: float) -> bool:
        if raw_intensity >= self.min_intensity:
            return True
        if self._last_evidence is None:
            return False
        ref = self._tbd_cached_ref
        peak_gate = self._tbd_cached_peak_gate
        return evidence_sum >= 0.55 * ref or peak_value >= peak_gate

    def detect_objects(self, binary_frame, original_frame):
        del binary_frame
        detections = []
        if self._last_evidence is None:
            return detections

        e = self._last_evidence
        foot = max(
            self.tbd_evidence_floor,
            float(np.percentile(e, self.tbd_evidence_percentile)),
        )
        k = 2 * self.tbd_peak_min_distance + 1
        local_max = maximum_filter(e, size=k)
        peak_mask = (e >= foot) & (e == local_max)
        ys, xs = np.where(peak_mask)

        for y, x in zip(ys.tolist(), xs.tolist()):
            r = self.tbd_local_patch_radius
            h, w = e.shape
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            patch = e[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            total = float(np.sum(patch))
            if total <= 1e-9:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            cx = float(np.sum(xx * patch) / total)
            cy = float(np.sum(yy * patch) / total)
            xi, yi = int(round(cx)), int(round(cy))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            intensity = float(original_frame[yi, xi])
            peak_value = float(e[yi, xi])
            if not self._passes_birth_gate(intensity, total, peak_value):
                continue
            detections.append({
                'centroid': (int(round(cx)), int(round(cy))),
                'intensity': intensity,
                'area': 1,
                'bbox': (xi - 1, yi - 1, 3, 3),
                'soft_evidence': peak_value,
                'evidence_sum': total,
                'proposal_score': 0.65 * peak_value + 0.35 * (total / max(1, patch.size)),
            })

        if len(detections) > self.tbd_max_proposals_per_frame:
            detections.sort(key=lambda det: det['proposal_score'], reverse=True)
            detections = detections[:self.tbd_max_proposals_per_frame]

        return detections

    def update_components(self, detections):
        n_components = len(self.bernoulli_components)
        n_detections = len(detections)

        if n_components == 0:
            for det in detections:
                pos = det['centroid']
                state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
                P = np.eye(4, dtype=np.float64) * 100.0
                component = BernoulliComponent(0.1, state, P, track_id=None)
                self.bernoulli_components.append(component)
            return

        if n_detections == 0:
            for component in self.bernoulli_components:
                component.r = (1 - self.detection_prob) * component.r
                component.consecutive_detections = 0
            return

        likelihood_matrix = np.zeros((n_components, n_detections))
        valid_associations = np.ones((n_components, n_detections), dtype=bool)

        for i, component in enumerate(self.bernoulli_components):
            for j, det in enumerate(detections):
                if component.check_physical_constraints(
                    det['centroid'],
                    self.max_acceleration,
                    self.max_direction_change,
                    self.max_speed,
                ):
                    base = component.likelihood(det['centroid'])
                    scale = self._evidence_likelihood_scale(component, det['centroid'])
                    likelihood_matrix[i, j] = base * scale
                else:
                    valid_associations[i, j] = False
                    likelihood_matrix[i, j] = 1e-10

        clutter_intensity = self.clutter_rate / (128 * 128)
        updated_components = []
        used_detections = set()

        for i, component in enumerate(self.bernoulli_components):
            r_miss = (1 - self.detection_prob) * component.r
            best_j = -1
            best_r = r_miss
            best_component = _clone_bernoulli_like(component, r_override=r_miss)
            best_component.consecutive_detections = 0

            for j, det in enumerate(detections):
                if j in used_detections or not valid_associations[i, j]:
                    continue

                likelihood = likelihood_matrix[i, j]
                r_update = (self.detection_prob * component.r * likelihood) / (
                    self.detection_prob * component.r * likelihood
                    + clutter_intensity * (1 - component.r)
                    + 1e-10
                )
                if r_update > best_r and r_update > 0.1:
                    best_r = r_update
                    best_j = j

            if best_j >= 0:
                component_copy = _clone_bernoulli_like(component)
                component_copy.update(detections[best_j]['centroid'], detections[best_j]['intensity'])
                component_copy.r = best_r
                updated_components.append(component_copy)
                used_detections.add(best_j)
            else:
                updated_components.append(best_component)

        for j, det in enumerate(detections):
            if j not in used_detections:
                pos = det['centroid']
                state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
                P = np.eye(4, dtype=np.float64) * 100.0
                component = BernoulliComponent(0.15, state, P, track_id=None)
                component.update(pos, det['intensity'])
                updated_components.append(component)

        self.bernoulli_components = updated_components
