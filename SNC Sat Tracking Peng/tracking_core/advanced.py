import cv2
import numpy as np
from collections import defaultdict, deque
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import mahalanobis
import scipy.stats as stats


class AdaptiveKalmanFilter2D:
    """
    Adaptive 2D Kalman Filter with innovation-based noise adjustment.
    """
    
    def __init__(self, initial_position, dt=1.0, process_noise=5.0, measurement_noise=2.0):
        self.dt = dt
        self.state = np.array([initial_position[0], initial_position[1], 0.0, 0.0], dtype=np.float64)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        # Observation matrix
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        
        # Process noise covariance
        q = process_noise
        self.Q = np.array([
            [q*dt**4/4, 0, q*dt**3/2, 0],
            [0, q*dt**4/4, 0, q*dt**3/2],
            [q*dt**3/2, 0, q*dt**2, 0],
            [0, q*dt**3/2, 0, q*dt**2]
        ], dtype=np.float64)
        
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float64) * measurement_noise
        
        # State covariance
        self.P = np.eye(4, dtype=np.float64) * 100.0
        
        # Innovation tracking
        self.innovation = np.zeros(2)
        self.innovation_history = deque(maxlen=10)
        self.S = np.eye(2)  # Innovation covariance
        
        # Adaptive parameters
        self.base_process_noise = process_noise
        self.base_measurement_noise = measurement_noise
        
    def predict(self):
        """Predict next state."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_position()
    
    def update(self, measurement):
        """Update state with measurement and adapt noise parameters."""
        z = np.array([measurement[0], measurement[1]], dtype=np.float64)
        
        # Innovation
        y = z - self.H @ self.state
        self.innovation = y
        
        # Innovation covariance
        self.S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        try:
            K = self.P @ self.H.T @ np.linalg.inv(self.S)
        except np.linalg.LinAlgError:
            K = self.P @ self.H.T @ np.linalg.pinv(self.S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # Track innovation for adaptive noise
        self.innovation_history.append(np.linalg.norm(y))
        
        # Adaptive noise: allow gentle shrink when stable; do not inflate Q on bad
        # innovations (that widens Mahalanobis gates and causes jumpy associations).
        if len(self.innovation_history) >= 5:
            avg_innovation = np.mean(self.innovation_history)
            if avg_innovation < 2.0:
                self.Q *= 0.99

        return self.get_position()
    
    def get_position(self):
        return (int(round(self.state[0])), int(round(self.state[1])))
    
    def get_position_float(self):
        return (self.state[0], self.state[1])
    
    def get_velocity(self):
        return (self.state[2], self.state[3])
    
    def get_speed(self):
        return np.sqrt(self.state[2]**2 + self.state[3]**2)
    
    def get_innovation_magnitude(self):
        return np.linalg.norm(self.innovation)
    
    def get_mahalanobis_distance(self, measurement):
        """Calculate Mahalanobis distance for gating."""
        z = np.array([measurement[0], measurement[1]], dtype=np.float64)
        predicted_z = self.H @ self.state
        diff = z - predicted_z
        
        try:
            S_inv = np.linalg.inv(self.S)
            dist = np.sqrt(diff.T @ S_inv @ diff)
        except np.linalg.LinAlgError:
            # Fall back to Euclidean if singular
            dist = np.linalg.norm(diff)
            
        return dist


def confidence_to_color(confidence):
    """Convert confidence (0-1) to BGR: Red -> Yellow -> Green"""
    confidence = max(0.0, min(1.0, confidence))
    if confidence < 0.5:
        t = confidence * 2
        return (0, int(255 * t), 255)
    else:
        t = (confidence - 0.5) * 2
        return (0, 255, int(255 * (1 - t)))


class AdvancedSatelliteTracker:
    """
    Advanced satellite tracking with intensity-based filtering and 
    optimized detection for small, faint objects.
    """
    
    def __init__(
        self,
        input_video_path,
        output_video_path,
        
        # === PREPROCESSING ===
        bg_threshold=8,
        bg_method='median',
        bg_sample_rate=50,  # Use every Nth frame for background
        use_temporal_smoothing=False,
        temporal_window=3,
        use_morphology=False,
        morph_kernel_size=3,
        
        # === DETECTION ===
        cluster_eps=3,
        cluster_min_samples=1,
        min_detection_area=1,
        max_detection_area=25,
        min_intensity=25,  # Minimum brightness for real satellites
        
        # === TRACKING ===
        max_distance=35,
        mahalanobis_threshold=4.0,
        process_noise=10.0,
        measurement_noise=2.0,
        track_timeout=15,
        lost_track_timeout=30,
        min_confidence_frames=3,
        max_confidence_frames=12,
        track_history_length=10,
        
        # === MOTION CONSTRAINTS ===
        min_track_length=4,
        max_acceleration=35.0,
        max_direction_change=100.0,
        use_kalman_display=True,
        velocity_gate_factor=2.0,
        min_speed=0.5,
        max_speed=50.0,
        
        # === QUALITY CONTROL ===
        # === QUALITY CONTROL ===
        min_detection_confidence=0.15,
        min_display_confidence=0.5,  # ★ ADD THIS LINE
        # === FRAME RANGE ===
        start_frame=0,
        end_frame=None,
        
        # === OUTPUT ===
        save_track_log=True,
        track_log_path=None,
        show_binary=False,
        verbose=True
    ):
        # Paths
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.verbose = verbose
        
        # Preprocessing
        self.bg_threshold = bg_threshold
        self.bg_method = bg_method
        self.bg_sample_rate = bg_sample_rate
        self.use_temporal_smoothing = use_temporal_smoothing
        self.temporal_window = temporal_window
        self.use_morphology = use_morphology
        self.morph_kernel_size = morph_kernel_size
        
        # Detection
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.min_detection_area = min_detection_area
        self.max_detection_area = max_detection_area
        self.min_intensity = min_intensity
        
        # Tracking
        self.max_distance = max_distance
        self.mahalanobis_threshold = mahalanobis_threshold
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.track_timeout = track_timeout
        self.lost_track_timeout = lost_track_timeout
        self.min_confidence_frames = min_confidence_frames
        self.max_confidence_frames = max_confidence_frames
        self.track_history_length = track_history_length
        
        # Motion
        self.min_track_length = min_track_length
        self.max_acceleration = max_acceleration
        self.max_direction_change = np.radians(max_direction_change)
        self.use_kalman_display = use_kalman_display
        self.velocity_gate_factor = velocity_gate_factor
        self.min_speed = min_speed
        self.max_speed = max_speed
        
        # Quality
        # Quality
        self.min_detection_confidence = min_detection_confidence
        self.min_display_confidence = min_display_confidence  # ★ ADD THIS LINE # ★ ADD THIS LINE
        
        # Frame range
        self.start_frame = start_frame
        self.end_frame = end_frame
        
        # Output
        self.save_track_log_enabled = save_track_log  # Fixed naming conflict
        if track_log_path is None:
            self.track_log_path = output_video_path.replace('.mp4', '_tracks.txt').replace('.avi', '_tracks.txt')
        else:
            self.track_log_path = track_log_path
        self.show_binary = show_binary
        
        # Internal state
        self.track_history = defaultdict(lambda: deque(maxlen=track_history_length))
        self.track_colors = defaultdict(lambda: deque(maxlen=track_history_length))
        self.velocity_history = defaultdict(lambda: deque(maxlen=10))
        self.next_track_id = 1
        self.active_tracks = {}
        self.kalman_filters = {}
        self.lost_tracks = {}
        self.background = None
        self.frame_buffer = deque(maxlen=temporal_window)
        
    def compute_background(self, frames):
        """Compute robust background model from sampled frames."""
        if self.verbose:
            print(f"Computing background from {len(frames)} frames using {self.bg_method} method...")
        
        stacked_frames = np.stack(frames, axis=0)
        
        if self.bg_method == 'min':
            # Minimum projection - good for bright objects
            background = np.min(stacked_frames, axis=0)
        elif self.bg_method == 'statistical' and len(frames) < 100:
            background = stats.mode(stacked_frames, axis=0, keepdims=False)[0]
        else:
            # Median is most robust
            background = np.median(stacked_frames, axis=0)
        
        return background.astype(np.uint8)
    
    def preprocess_frame(self, frame):
        """Minimal preprocessing optimized for small point sources."""
        
        # Temporal smoothing (optional)
        if self.use_temporal_smoothing and len(self.frame_buffer) == self.temporal_window:
            frame_smoothed = np.mean(list(self.frame_buffer), axis=0).astype(np.uint8)
        else:
            frame_smoothed = frame
        
        # Background subtraction
        diff = cv2.absdiff(frame_smoothed, self.background)
        
        # Very light blur to reduce single-pixel noise
        diff = cv2.GaussianBlur(diff, (3, 3), 0.3)
        
        # Threshold
        _, binary = cv2.threshold(diff, self.bg_threshold, 255, cv2.THRESH_BINARY)
        
        # Optional morphological filtering (minimal)
        if self.use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.morph_kernel_size, self.morph_kernel_size))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return binary
    
    def detect_objects(self, binary_frame, original_frame):
        """
        Detect objects with intensity-based filtering.
        This filters out noise by checking brightness in the original frame.
        """
        detections = []
        white_pixels = np.where(binary_frame == 255)
        
        if len(white_pixels[0]) == 0:
            return detections
        
        coords = list(zip(white_pixels[1], white_pixels[0]))
        
        if len(coords) == 0:
            return detections
        
        if len(coords) == 1:
            x, y = coords[0]
            # Check brightness in original frame
            intensity = float(original_frame[y, x])
            if intensity >= self.min_intensity:  # Real satellites should be bright
                detections.append({
                    'centroid': (x, y),
                    'bbox': (x-1, y-1, 3, 3),
                    'area': 1,
                    'intensity': intensity
                })
        else:
            clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples).fit(coords)
            
            for cluster_id in set(clustering.labels_):
                if cluster_id == -1:
                    continue
                    
                cluster_coords = [coords[i] for i in range(len(coords)) 
                                if clustering.labels_[i] == cluster_id]
                
                if len(cluster_coords) > 0:
                    area = len(cluster_coords)
                    
                    # Area filtering
                    if area < self.min_detection_area or area > self.max_detection_area:
                        continue
                    
                    x_coords = [c[0] for c in cluster_coords]
                    y_coords = [c[1] for c in cluster_coords]
                    cx, cy = int(np.mean(x_coords)), int(np.mean(y_coords))
                    
                    # Calculate intensity statistics in original frame
                    intensities = [float(original_frame[y, x]) for x, y in cluster_coords]
                    avg_intensity = np.mean(intensities)
                    max_intensity = np.max(intensities)
                    
                    # Filter by brightness - real satellites should be bright
                    if max_intensity >= self.min_intensity:
                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)
                        
                        detections.append({
                            'centroid': (cx, cy),
                            'bbox': (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
                            'area': area,
                            'intensity': avg_intensity,
                            'max_intensity': max_intensity
                        })
        
        return detections
    
    def calculate_track_confidence(self, track):
        """Calculate track confidence score."""
        consecutive = track['consecutive_detections']
        total = track.get('total_detections', consecutive)
        age = track['age']
        
        # Base confidence from consecutive detections
        if consecutive < self.min_confidence_frames:
            base = 0.2 + (consecutive / self.min_confidence_frames) * 0.3
        else:
            effective = min(consecutive, self.max_confidence_frames)
            base = 0.5 + 0.5 * (effective - self.min_confidence_frames) / \
                   (self.max_confidence_frames - self.min_confidence_frames)
        
        # Detection rate bonus
        if total > 0:
            detection_rate = consecutive / total
            base *= (0.7 + 0.3 * detection_rate)
        
        # Age penalty (tracks that miss too often)
        if age > consecutive:
            age_penalty = min(0.3, (age - consecutive) * 0.05)
            base -= age_penalty
        
        # Intensity bonus - brighter objects are more likely real
        if 'avg_intensity' in track:
            intensity_bonus = min(0.1, track['avg_intensity'] / 500)
            base += intensity_bonus
        
        # Speed consistency bonus
        track_id = track['id']
        if track_id in self.velocity_history and len(self.velocity_history[track_id]) >= 3:
            speeds = [np.sqrt(v[0]**2 + v[1]**2) for v in self.velocity_history[track_id]]
            speed_std = np.std(speeds)
            if speed_std < 2.0:  # Consistent speed
                base += 0.05
        
        return max(0.0, min(1.0, base))
    
    def check_motion_validity(self, track_id, new_detection):
        """Comprehensive motion consistency check."""
        if track_id not in self.kalman_filters:
            return True
        
        kf = self.kalman_filters[track_id]
        track = self.active_tracks[track_id]
        
        # Get current motion state
        current_vel = kf.get_velocity()
        current_speed = kf.get_speed()
        last_pos = track.get('last_raw_centroid', track['centroid'])
        new_pos = new_detection['centroid']
        
        # Calculate implied velocity
        new_vel = (new_pos[0] - last_pos[0], new_pos[1] - last_pos[1])
        new_speed = np.sqrt(new_vel[0]**2 + new_vel[1]**2)
        
        # Check 1: Speed limits
        if new_speed > self.max_speed:
            return False

        # Young tracks: cap per-frame step to reduce cross-screen snaps
        if track['consecutive_detections'] < 3:
            step_cap = self.max_distance * 1.05 + 2.0
            if new_speed > step_cap:
                return False
        
        # Check 2: Acceleration constraint (only for established tracks)
        if track['consecutive_detections'] >= 3 and current_speed > self.min_speed:
            accel = np.sqrt((new_vel[0] - current_vel[0])**2 + 
                          (new_vel[1] - current_vel[1])**2)
            if accel > self.max_acceleration:
                return False
        
        # Check 3: Direction consistency (only for fast-moving tracks)
        if track['consecutive_detections'] >= 3 and current_speed > 2.0 and new_speed > 2.0:
            # Calculate angle between current and new velocity
            dot_product = current_vel[0] * new_vel[0] + current_vel[1] * new_vel[1]
            cos_angle = dot_product / (current_speed * new_speed + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_diff = np.arccos(cos_angle)
            
            if angle_diff > self.max_direction_change:
                return False
        
        return True

    def association_match_acceptable(self, track_id, detection, raw_cost):
        """Whether a (track, detection) pair passes gates (Hungarian / greedy matchers)."""
        if raw_cost >= 1e8 - 1:
            return False
        det_pos = detection['centroid']
        if track_id in self.kalman_filters:
            kf = self.kalman_filters[track_id]
            speed = kf.get_speed()
            thr = min(
                self.mahalanobis_threshold * (1.0 + speed / 12.0),
                self.mahalanobis_threshold * 1.35,
            )
            pred = np.array(kf.get_position_float(), dtype=np.float64)
            dvec = np.array(det_pos, dtype=np.float64)
            pix = float(np.linalg.norm(dvec - pred))
            pix_cap = self.max_distance + self.velocity_gate_factor * speed + 6.0
            if raw_cost > thr or pix > pix_cap:
                return False
            return self.check_motion_validity(track_id, detection)
        return raw_cost < self.max_distance

    def compute_association_costs(self, detections):
        """Compute cost matrix using Mahalanobis distance."""
        if not self.active_tracks or not detections:
            return None, [], []
        
        track_ids = list(self.active_tracks.keys())
        cost_matrix = np.full((len(track_ids), len(detections)), 1e9)
        
        for i, track_id in enumerate(track_ids):
            track = self.active_tracks[track_id]
            kf = self.kalman_filters.get(track_id)
            
            if kf is not None:
                # Use Mahalanobis distance for gating
                for j, det in enumerate(detections):
                    det_pos = det['centroid']
                    
                    # Calculate Mahalanobis distance
                    mahal_dist = kf.get_mahalanobis_distance(det_pos)
                    
                    # Adaptive gating based on velocity
                    speed = kf.get_speed()
                    adaptive_threshold = min(
                        self.mahalanobis_threshold * (1.0 + speed / 12.0),
                        self.mahalanobis_threshold * 1.35,
                    )
                    pred = np.array(kf.get_position_float(), dtype=np.float64)
                    det_vec = np.array(det_pos, dtype=np.float64)
                    pixel_dist = float(np.linalg.norm(det_vec - pred))
                    pixel_cap = self.max_distance + self.velocity_gate_factor * speed + 6.0

                    if mahal_dist < adaptive_threshold and pixel_dist <= pixel_cap:
                        if self.check_motion_validity(track_id, det):
                            cost_matrix[i, j] = mahal_dist
            else:
                # Fallback to Euclidean for new tracks
                last_pos = track.get('last_raw_centroid', track['centroid'])
                for j, det in enumerate(detections):
                    det_pos = det['centroid']
                    dist = np.sqrt((last_pos[0] - det_pos[0])**2 + 
                                 (last_pos[1] - det_pos[1])**2)
                    if dist < self.max_distance:
                        cost_matrix[i, j] = dist
        
        return cost_matrix, track_ids, detections
    
    def create_new_track(self, detection):
        """Initialize a new track."""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        pos = detection['centroid']
        track = {
            'id': track_id,
            'centroid': pos,
            'last_raw_centroid': pos,
            'kalman_centroid': pos,
            'first_position': pos,
            'bbox': detection['bbox'],
            'area': detection['area'],
            'age': 1,
            'consecutive_detections': 1,
            'confidence': 0.2,
            'total_detections': 1,
            'max_displacement': 0.0,
            'birth_frame': 0,
            'avg_intensity': detection.get('intensity', 0)
        }
        
        # Initialize Kalman filter
        self.kalman_filters[track_id] = AdaptiveKalmanFilter2D(
            initial_position=pos,
            dt=1.0,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise
        )
        
        self.active_tracks[track_id] = track
        return track
    
    def try_reidentify_lost_tracks(self, detections):
        """Attempt to re-identify lost tracks."""
        reidentified = []
        used_dets = set()
        
        for track_id, lost_info in list(self.lost_tracks.items()):
            last_pos = lost_info['last_centroid']
            velocity = lost_info['velocity']
            lost_age = lost_info['lost_age']
            
            # Predict position
            predicted_pos = (last_pos[0] + velocity[0] * lost_age,
                           last_pos[1] + velocity[1] * lost_age)
            
            best_det_idx = -1
            best_dist = self.max_distance * 0.95
            reid_cap = min(
                self.max_distance + lost_age * self.max_speed * 0.35,
                self.max_distance * 1.25,
            )

            for j, det in enumerate(detections):
                if j in used_dets:
                    continue
                dist = np.sqrt((predicted_pos[0] - det['centroid'][0])**2 +
                             (predicted_pos[1] - det['centroid'][1])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_det_idx = j

            if best_det_idx >= 0 and best_dist <= reid_cap:
                det = detections[best_det_idx]
                used_dets.add(best_det_idx)
                
                # Reinitialize track
                track = {
                    'id': track_id,
                    'centroid': det['centroid'],
                    'last_raw_centroid': det['centroid'],
                    'kalman_centroid': det['centroid'],
                    'first_position': lost_info.get('first_position', det['centroid']),
                    'bbox': det['bbox'],
                    'area': det['area'],
                    'age': 1,
                    'consecutive_detections': 1,
                    'confidence': 0.4,
                    'total_detections': lost_info.get('total_detections', 1) + 1,
                    'max_displacement': lost_info.get('max_displacement', 0.0),
                    'birth_frame': lost_info.get('birth_frame', 0),
                    'avg_intensity': det.get('intensity', 0)
                }
                
                # Reinitialize Kalman filter with velocity
                self.kalman_filters[track_id] = AdaptiveKalmanFilter2D(
                    initial_position=det['centroid'],
                    dt=1.0,
                    process_noise=self.process_noise,
                    measurement_noise=self.measurement_noise
                )
                self.kalman_filters[track_id].state[2] = velocity[0]
                self.kalman_filters[track_id].state[3] = velocity[1]
                
                self.active_tracks[track_id] = track
                reidentified.append(track)
                del self.lost_tracks[track_id]
        
        return reidentified, used_dets
    
    def associate_tracks(self, detections, frame_idx):
        """Main tracking association logic."""
        
        # Predict all tracks
        for track_id in self.active_tracks:
            if track_id in self.kalman_filters:
                self.kalman_filters[track_id].predict()
        
        # Update lost tracks
        lost_to_remove = [tid for tid, info in self.lost_tracks.items()
                         if info['lost_age'] > self.lost_track_timeout]
        for tid in lost_to_remove:
            del self.lost_tracks[tid]
        for tid in self.lost_tracks:
            self.lost_tracks[tid]['lost_age'] += 1
        
        # Handle no detections
        if not detections:
            for track in self.active_tracks.values():
                track['age'] += 1
                track['confidence'] = max(0.0, track.get('confidence', 0.5) - 0.1)
                if track['id'] in self.kalman_filters:
                    track['kalman_centroid'] = self.kalman_filters[track['id']].get_position()
            return list(self.active_tracks.values())
        
        # Handle no active tracks
        if not self.active_tracks:
            reidentified, used_dets = self.try_reidentify_lost_tracks(detections)
            tracks = reidentified
            for j, det in enumerate(detections):
                if j not in used_dets:
                    new_track = self.create_new_track(det)
                    new_track['birth_frame'] = frame_idx
                    tracks.append(new_track)
            return tracks
        
        # Compute association costs
        cost_matrix, track_ids, _ = self.compute_association_costs(detections)
        if cost_matrix is None:
            return []
        
        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        tracks = []
        used_detections = set()
        matched_tracks = set()
        
        # Process matches (Mahalanobis costs are not comparable to max_distance*2)
        for row_idx, col_idx in zip(row_indices, col_indices):
            track_id = track_ids[row_idx]
            detection = detections[col_idx]
            det_pos = detection['centroid']
            raw_cost = cost_matrix[row_idx, col_idx]
            if not self.association_match_acceptable(track_id, detection, raw_cost):
                continue

            # Update Kalman filter
            if track_id in self.kalman_filters:
                filtered_pos = self.kalman_filters[track_id].update(det_pos)
                velocity = self.kalman_filters[track_id].get_velocity()
                self.velocity_history[track_id].append(velocity)
            else:
                filtered_pos = det_pos

            # Update track
            first_pos = self.active_tracks[track_id].get('first_position', det_pos)
            displacement = np.sqrt((det_pos[0] - first_pos[0])**2 +
                                     (det_pos[1] - first_pos[1])**2)

            # Update average intensity
            old_intensity = self.active_tracks[track_id].get('avg_intensity', 0)
            new_intensity = detection.get('intensity', old_intensity)
            total_dets = self.active_tracks[track_id].get('total_detections', 1)
            avg_intensity = (old_intensity * total_dets + new_intensity) / (total_dets + 1)

            self.active_tracks[track_id].update({
                'centroid': det_pos,
                'last_raw_centroid': det_pos,
                'kalman_centroid': filtered_pos,
                'bbox': detection['bbox'],
                'area': detection['area'],
                'age': 1,
                'consecutive_detections': self.active_tracks[track_id]['consecutive_detections'] + 1,
                'total_detections': self.active_tracks[track_id].get('total_detections', 1) + 1,
                'max_displacement': max(self.active_tracks[track_id].get('max_displacement', 0),
                                        displacement),
                'avg_intensity': avg_intensity
            })

            self.active_tracks[track_id]['confidence'] = self.calculate_track_confidence(
                self.active_tracks[track_id])

            tracks.append(self.active_tracks[track_id])
            used_detections.add(col_idx)
            matched_tracks.add(track_id)
        
        # Handle unmatched tracks
        for track_id in track_ids:
            if track_id not in matched_tracks:
                self.active_tracks[track_id]['age'] += 1
                self.active_tracks[track_id]['confidence'] = max(0.0,
                    self.active_tracks[track_id].get('confidence', 0.5) - 0.1)
                if track_id in self.kalman_filters:
                    self.active_tracks[track_id]['kalman_centroid'] = \
                        self.kalman_filters[track_id].get_position()
                tracks.append(self.active_tracks[track_id])
        
        # Try to reidentify lost tracks
        unmatched_dets = [detections[j] for j in range(len(detections)) 
                         if j not in used_detections]
        reidentified, reident_used = self.try_reidentify_lost_tracks(unmatched_dets)
        
        for track in reidentified:
            tracks.append(track)
        
        # Create new tracks from remaining detections
        for j, det in enumerate(unmatched_dets):
            if j not in reident_used:
                new_track = self.create_new_track(det)
                new_track['birth_frame'] = frame_idx
                tracks.append(new_track)
        
        # Move timed-out tracks to lost
        tracks_to_remove = []
        for track_id, track in self.active_tracks.items():
            if track['age'] > self.track_timeout:
                tracks_to_remove.append(track_id)
                self.lost_tracks[track_id] = {
                    'last_centroid': track.get('last_raw_centroid', track['centroid']),
                    'velocity': self.kalman_filters[track_id].get_velocity() 
                                if track_id in self.kalman_filters else (0, 0),
                    'lost_age': 0,
                    'total_detections': track.get('total_detections', 1),
                    'first_position': track.get('first_position'),
                    'max_displacement': track.get('max_displacement', 0),
                    'birth_frame': track.get('birth_frame', 0)
                }
        
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
            if track_id in self.kalman_filters:
                del self.kalman_filters[track_id]
        
        return tracks
    
    def is_valid_track(self, track):
        """
        Comprehensive track validation with smoothness and physics checks.
        """
        track_id = track['id']
        
        # 1. Confidence filter
        confidence = track.get('confidence', 0.0)
        if confidence < self.min_display_confidence:
            return False
        
        total_dets = track.get('total_detections', track.get('consecutive_detections', 1))
        
        # 2. Minimum length
        if total_dets < self.min_track_length:
            return False
        
        # 3. Kalman filter innovation check (prediction consistency)
        if track_id in self.kalman_filters:
            kf = self.kalman_filters[track_id]
            if len(kf.innovation_history) >= 5:
                avg_innovation = np.mean(kf.innovation_history)
                if avg_innovation > 15.0:  # Poor predictions = bad track
                    return False
        
        # 4. Trajectory smoothness check
        if track_id in self.track_history and len(self.track_history[track_id]) >= 4:
            smoothness = self.calculate_trajectory_smoothness(track_id)
            if smoothness < 0.35:  # Reject jumpy trajectories
                return False
        
        # 5. Direction consistency check
        if track_id in self.track_history and len(self.track_history[track_id]) >= 5:
            if not self.check_direction_consistency(track_id):
                return False
        
        # 6. Motion requirement (must be moving)
        if track_id in self.kalman_filters:
            speed = self.kalman_filters[track_id].get_speed()
            if speed >= self.min_speed:
                return True
        
        # 7. Displacement requirement
        displacement = track.get('max_displacement', 0)
        if total_dets > 3 and displacement > self.min_speed * total_dets * 0.5:
            return True
        
        return False
    
    def annotate_frame(self, frame, tracks):
        """Annotate frame with tracks (no ID labels, shorter trails)."""
        annotated = frame.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
        
        for track in tracks:
            if not self.is_valid_track(track):
                continue
            
            track_id = track['id']
            
            # Choose position
            if self.use_kalman_display:
                centroid = track.get('kalman_centroid', track['centroid'])
            else:
                centroid = track['centroid']
            
            bbox = track['bbox']
            confidence = track.get('confidence', 0.5)
            color = confidence_to_color(confidence)
            
            # Store history
            self.track_history[track_id].append(centroid)
            self.track_colors[track_id].append(color)
            
            # Draw bounding box
            x, y, w, h = bbox
            cv2.rectangle(annotated, (x-2, y-2), (x + w + 2, y + h + 2), color, 1)
            
            # Draw trajectory (trails)
            if len(self.track_history[track_id]) > 1:
                points = list(self.track_history[track_id])
                colors = list(self.track_colors[track_id])
                for i in range(len(points) - 1):
                    cv2.line(annotated, points[i], points[i + 1], colors[i + 1], 1)
            
            # Draw centroid
            cv2.circle(annotated, centroid, 2, color, -1)
            
            # NO track ID label drawn
        
        return annotated
    def calculate_trajectory_smoothness(self, track_id):
        """
        Calculate smoothness score based on acceleration consistency.
        Returns 0.0 (very jumpy) to 1.0 (very smooth).
        """
        if track_id not in self.track_history:
            return 0.5
        
        history = list(self.track_history[track_id])
        if len(history) < 4:
            return 0.5
        
        # Calculate accelerations between consecutive position changes
        accelerations = []
        for i in range(len(history) - 2):
            p0, p1, p2 = history[i], history[i+1], history[i+2]
            
            # Velocities
            v1 = (p1[0] - p0[0], p1[1] - p0[1])
            v2 = (p2[0] - p1[0], p2[1] - p1[1])
            
            # Acceleration (change in velocity)
            accel = np.sqrt((v2[0] - v1[0])**2 + (v2[1] - v1[1])**2)
            accelerations.append(accel)
        
        if not accelerations:
            return 0.5
        
        # Calculate statistics
        mean_accel = np.mean(accelerations)
        std_accel = np.std(accelerations)
        max_accel = np.max(accelerations)
        
        # Smooth trajectories have low mean acceleration and low variance
        smoothness = 0.0
        
        # Penalize high mean acceleration
        if mean_accel < 2.0:
            smoothness += 0.4
        elif mean_accel < 5.0:
            smoothness += 0.2
        
        # Penalize high variance (erratic motion)
        if std_accel < 2.0:
            smoothness += 0.4
        elif std_accel < 5.0:
            smoothness += 0.2
        
        # Penalize any single large jump
        if max_accel < 10.0:
            smoothness += 0.2
        elif max_accel < 20.0:
            smoothness += 0.1
        
        return min(1.0, smoothness)

    def check_direction_consistency(self, track_id):
        """
        Check if track maintains a consistent overall direction.
        Returns True if consistent, False if erratic.
        """
        if track_id not in self.track_history:
            return True
        
        history = list(self.track_history[track_id])
        if len(history) < 5:
            return True
        
        # Overall direction (first to last point)
        overall_vec = (history[-1][0] - history[0][0], 
                    history[-1][1] - history[0][1])
        overall_angle = np.arctan2(overall_vec[1], overall_vec[0])
        
        # Check each segment's alignment with overall direction
        misalignments = 0
        for i in range(len(history) - 1):
            segment_vec = (history[i+1][0] - history[i][0],
                        history[i+1][1] - history[i][1])
            segment_angle = np.arctan2(segment_vec[1], segment_vec[0])
            
            # Angle difference
            angle_diff = abs(segment_angle - overall_angle)
            # Normalize to [0, pi]
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
            
            # If segment deviates more than 90 degrees from overall direction
            if angle_diff > np.pi/2:
                misalignments += 1
        
        # Allow some flexibility, but not too much
        max_allowed_misalignments = max(1, len(history) // 4)
        return misalignments <= max_allowed_misalignments
        
    def save_track_log(self, track_log):
        """Save tracking log to file."""
        with open(self.track_log_path, 'w') as f:
            f.write("# Advanced Satellite Tracking Log\n")
            f.write(f"# Input: {self.input_video_path}\n")
            f.write(f"# Output: {self.output_video_path}\n")
            f.write("#\n")
            f.write("# frame, track_id, x, y, confidence, speed, vx, vy, quality, intensity\n")
            f.write("#" + "=" * 80 + "\n")
            
            for entry in track_log:
                f.write(f"{entry['frame']}, {entry['track_id']}, "
                       f"{entry['x']}, {entry['y']}, "
                       f"{entry['confidence']:.4f}, {entry['speed']:.4f}, "
                       f"{entry['vx']:.4f}, {entry['vy']:.4f}, "
                       f"{entry['quality']:.4f}, {entry['intensity']:.2f}\n")
        
        if self.verbose:
            print(f"   📝 Track log saved: {self.track_log_path}")
    
    def run(self):
        """Execute the tracking pipeline."""
        if self.verbose:
            print("=" * 80)
            print("🛰️  ADVANCED SATELLITE TRACKING SYSTEM")
            print("=" * 80)
            print(f"\n📹 Input:  {self.input_video_path}")
            print(f"📹 Output: {self.output_video_path}")
            
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
        
        # Frame range
        start = self.start_frame
        end = self.end_frame if self.end_frame else len(all_frames)
        end = min(end, len(all_frames))
        frames_to_process = all_frames[start:end]
        
        if self.verbose:
            print(f"   Processing frames {start} to {end} ({len(frames_to_process)} frames)")
        
        # Compute background from sampled frames
        sample_frames = all_frames[::self.bg_sample_rate]
        self.background = self.compute_background(sample_frames)
        
        if self.verbose:
            print(f"\n⚙️  TRACKING PARAMETERS:")
            print(f"   Background threshold:     {self.bg_threshold}")
            print(f"   Min intensity:            {self.min_intensity}")
            print(f"   Max distance:             {self.max_distance}")
            print(f"   Mahalanobis threshold:    {self.mahalanobis_threshold}")
            print(f"   Process noise:            {self.process_noise}")
            print(f"   Measurement noise:        {self.measurement_noise}")
            print(f"   Max acceleration:         {self.max_acceleration}")
            print(f"   Max direction change:     {np.degrees(self.max_direction_change):.1f}°")
            print(f"   Min track length:         {self.min_track_length}")
            print(f"   Min/Max speed:            {self.min_speed}/{self.max_speed}")
            print(f"   Track history length:     {self.track_history_length}")
            print(f"   Temporal smoothing:       {self.use_temporal_smoothing}")
            print("=" * 80)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if self.show_binary:
            out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width * 3, height))
        else:
            out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width * 2, height))
        
        # Reopen for processing
        cap = cv2.VideoCapture(self.input_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        total_detections = 0
        track_log = []
        
        if self.verbose:
            print("\n🎬 Processing frames...")
        
        for i, gray_frame in enumerate(frames_to_process):
            ret, original_frame = cap.read()
            if not ret:
                break
            
            # Add to temporal buffer
            self.frame_buffer.append(gray_frame)
            
            # Preprocess
            binary = self.preprocess_frame(gray_frame)
            
            # Detect with intensity filtering
            detections = self.detect_objects(binary, gray_frame)
            total_detections += len(detections)
            
            # Track
            tracks = self.associate_tracks(detections, start + i)
            
            # Log tracks
            if self.save_track_log_enabled:
                for track in tracks:
                    if self.is_valid_track(track):
                        track_id = track['id']
                        
                        if self.use_kalman_display:
                            centroid = track.get('kalman_centroid', track['centroid'])
                        else:
                            centroid = track['centroid']
                        
                        confidence = track.get('confidence', 0.5)
                        intensity = track.get('avg_intensity', 0)
                        
                        if track_id in self.kalman_filters:
                            speed = self.kalman_filters[track_id].get_speed()
                            vx, vy = self.kalman_filters[track_id].get_velocity()
                        else:
                            speed = 0.0
                            vx, vy = 0.0, 0.0
                        
                        # Quality metric
                        total_dets = track.get('total_detections', 1)
                        quality = confidence * min(1.0, total_dets / 10.0)
                        
                        track_log.append({
                            'frame': start + i,
                            'track_id': track_id,
                            'x': centroid[0],
                            'y': centroid[1],
                            'confidence': confidence,
                            'speed': speed,
                            'vx': vx,
                            'vy': vy,
                            'quality': quality,
                            'intensity': intensity
                        })
            
            # Annotate
            annotated = self.annotate_frame(original_frame, tracks)
            
            # Output
            if self.show_binary:
                binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                side_by_side = np.hstack([original_frame, binary_colored, annotated])
            else:
                side_by_side = np.hstack([original_frame, annotated])
            out.write(side_by_side)
            
            # Progress
            if self.verbose and ((i + 1) % 100 == 0 or i == len(frames_to_process) - 1):
                moving = sum(1 for t in self.active_tracks.values() if self.is_valid_track(t))
                print(f"   Frame {i+1}/{len(frames_to_process)} - "
                      f"Active: {len(self.active_tracks)}, "
                      f"Valid: {moving}, "
                      f"Detections: {len(detections)}")
        
        cap.release()
        out.release()
        
        # Save log
        if self.save_track_log_enabled and track_log:
            self.save_track_log(track_log)
        
        # Summary
        unique_tracks = len(self.track_history)
        unique_logged = len(set(e['track_id'] for e in track_log)) if track_log else 0
        
        if self.verbose:
            print(f"\n🎉 COMPLETE!")
            print(f"   Frames processed:  {len(frames_to_process)}")
            print(f"   Total detections:  {total_detections}")
            print(f"   Unique tracks:     {unique_tracks}")
            print(f"   Valid tracks:      {unique_logged}")
            print(f"   Log entries:       {len(track_log)}")
            print(f"   Output saved:      {self.output_video_path}")
            print("=" * 80)
        
        return {
            'frames_processed': len(frames_to_process),
            'total_detections': total_detections,
            'unique_tracks': unique_tracks,
            'valid_tracks': unique_logged,
            'log_entries': len(track_log)
        }
