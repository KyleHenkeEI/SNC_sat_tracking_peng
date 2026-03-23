import cv2
import numpy as np
from collections import defaultdict, deque
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal, poisson
import scipy.stats as stats
from copy import deepcopy


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
        verbose=True
    ):
        # Paths
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.verbose = verbose
        
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
        
        # Internal state
        self.bernoulli_components = []
        self.next_track_id = 1
        self.track_history = defaultdict(lambda: deque(maxlen=track_history_length))
        self.track_colors = defaultdict(lambda: deque(maxlen=track_history_length))
        self.background = None
        self.frame_buffer = deque(maxlen=3)
        
    def compute_background(self, frames):
        """Compute background model."""
        if self.verbose:
            print(f"Computing background from {len(frames)} frames...")
        sample_frames = frames[::self.bg_sample_rate]
        stacked = np.stack(sample_frames, axis=0)
        background = np.median(stacked, axis=0)
        return background.astype(np.uint8)
    
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
        
        # Compute background
        self.background = self.compute_background(all_frames)
        
        if self.verbose:
            print(f"\n⚙️  PARAMETERS:")
            print(f"   Existence threshold:     {self.existence_threshold}")
            print(f"   Min display confidence:  {self.min_display_confidence}")
            print(f"   Min track length:        {self.min_track_length}")
            print(f"   Max acceleration:        {self.max_acceleration}")
            print(f"   Max direction change:    {self.max_direction_change}°")
            print(f"   Min/Max speed:           {self.min_speed}/{self.max_speed}")
            print("=" * 80)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
            
            # Annotate
            annotated = self.annotate_frame(original_frame, tracks)
            
            # Output
            side_by_side = np.hstack([original_frame, annotated])
            out.write(side_by_side)
            
            # Progress
            if self.verbose and ((i + 1) % 100 == 0 or i == len(frames_to_process) - 1):
                n_components = len(self.bernoulli_components)
                n_confirmed = len(tracks)
                print(f"   Frame {i+1}/{len(frames_to_process)} - "
                      f"Components: {n_components}, "
                      f"Confirmed: {n_confirmed}, "
                      f"Detections: {len(detections)}")
        
        cap.release()
        out.release()
        
        # Save log
        if self.save_track_log_enabled and track_log:
            self.save_track_log(track_log)
        
        # Summary
        unique_tracks = len(set(e['track_id'] for e in track_log)) if track_log else 0
        
        if self.verbose:
            print(f"\n🎉 COMPLETE!")
            print(f"   Frames processed:  {len(frames_to_process)}")
            print(f"   Total detections:  {total_detections}")
            print(f"   Unique tracks:     {unique_tracks}")
            print(f"   Log entries:       {len(track_log)}")
            print(f"   Output saved:      {self.output_video_path}")
            print("=" * 80)
        
        return {
            'frames_processed': len(frames_to_process),
            'total_detections': total_detections,
            'unique_tracks': unique_tracks,
            'log_entries': len(track_log)
        }
