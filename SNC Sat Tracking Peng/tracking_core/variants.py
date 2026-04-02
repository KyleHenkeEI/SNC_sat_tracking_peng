"""Tracker variant classes and TRACKER_VARIANTS registry."""
from __future__ import annotations

import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

from tracking_core.advanced import AdvancedSatelliteTracker
from tracking_core.pmb import (
    BernoulliComponent,
    PoissonMultiBernoulliTracker,
    SparseBudgetPmTracker,
    TextbookPoissonMultiBernoulliTracker,
    TrueTbdPmTracker,
)
from tracking_core.pmbm import PoissonMultiBernoulliMixtureTracker
from tracking_core.presets import apply_preset_to_kwargs


ADVANCED_BASELINE_KWARGS = dict(
    bg_threshold=10,
    bg_method='median',
    bg_sample_rate=50,
    use_temporal_smoothing=False,
    temporal_window=3,
    use_morphology=False,
    morph_kernel_size=3,
    cluster_eps=3,
    cluster_min_samples=1,
    min_detection_area=1,
    max_detection_area=80,
    min_intensity=20,
    max_distance=22,
    mahalanobis_threshold=3.5,
    process_noise=8.0,
    measurement_noise=2.0,
    track_timeout=22,
    lost_track_timeout=45,
    min_confidence_frames=3,
    max_confidence_frames=12,
    track_history_length=10,
    min_track_length=4,
    max_acceleration=35.0,
    max_direction_change=70.0,
    use_kalman_display=True,
    velocity_gate_factor=2.0,
    min_speed=0.3,
    max_speed=60.0,
    min_detection_confidence=0.22,
    min_display_confidence=0.58,
    start_frame=0,
    end_frame=None,
    save_track_log=True,
    show_binary=False,
    verbose=True,
)

PMB_BASELINE_KWARGS = dict(
    bg_threshold=10,
    bg_sample_rate=50,
    use_temporal_smoothing=False,
    use_morphology=False,
    cluster_eps=3,
    cluster_min_samples=1,
    min_detection_area=1,
    max_detection_area=300,
    min_intensity=28,
    birth_rate=0.1,
    survival_prob=0.99,
    detection_prob=0.85,
    clutter_rate=5.0,
    existence_threshold=0.5,
    pruning_threshold=0.01,
    max_acceleration=10.0,
    max_direction_change=70.0,
    max_speed=300.0,
    min_speed=0.4,
    min_track_length=5,
    min_confidence_frames=3,
    max_confidence_frames=15,
    min_display_confidence=0.75,
    track_history_length=10,
    start_frame=0,
    end_frame=None,
    save_track_log=True,
    verbose=True,
    # Air-to-air (see AIR_TO_AIR_ADAPTATION.md); preset `air_to_air` enables for all PMB kind trackers
    a2a_phase_stabilize=False,
    a2a_clahe_clip_limit=0.0,
    a2a_clahe_tile_size=8,
    a2a_imu_json_path=None,
    a2a_imu_apply_compensation=False,
    a2a_imu_fx=900.0,
    a2a_imu_fy=900.0,
    a2a_imu_default_dt=1.0 / 30.0,
)

def _predict_active_tracks(tracker):
    for track_id in list(tracker.active_tracks.keys()):
        if track_id in tracker.kalman_filters:
            tracker.kalman_filters[track_id].predict()


def _refresh_lost_tracks(tracker):
    lost_to_remove = [
        tid for tid, info in tracker.lost_tracks.items()
        if info['lost_age'] > tracker.lost_track_timeout
    ]
    for tid in lost_to_remove:
        del tracker.lost_tracks[tid]
    for tid in tracker.lost_tracks:
        tracker.lost_tracks[tid]['lost_age'] += 1


def _calculate_confidence_for_tracker(tracker, track):
    if hasattr(tracker, 'calculate_track_confidence'):
        return tracker.calculate_track_confidence(track)
    return tracker.calculate_confidence(track)


def _update_track_from_detection(tracker, track_id, detection):
    det_pos = detection['centroid']

    if track_id in tracker.kalman_filters:
        filtered_pos = tracker.kalman_filters[track_id].update(det_pos)
        if hasattr(tracker, 'velocity_history'):
            velocity = tracker.kalman_filters[track_id].get_velocity()
            tracker.velocity_history[track_id].append(velocity)
    else:
        filtered_pos = det_pos

    first_pos = tracker.active_tracks[track_id].get('first_position', det_pos)
    displacement = np.sqrt((det_pos[0] - first_pos[0])**2 + (det_pos[1] - first_pos[1])**2)

    old_intensity = tracker.active_tracks[track_id].get('avg_intensity', 0.0)
    new_intensity = detection.get('intensity', old_intensity)
    total_dets = tracker.active_tracks[track_id].get('total_detections', 1)
    avg_intensity = (old_intensity * total_dets + new_intensity) / (total_dets + 1)

    tracker.active_tracks[track_id].update({
        'centroid': det_pos,
        'last_raw_centroid': det_pos,
        'kalman_centroid': filtered_pos,
        'bbox': detection.get('bbox', tracker.active_tracks[track_id].get('bbox', (det_pos[0]-1, det_pos[1]-1, 3, 3))),
        'area': detection.get('area', tracker.active_tracks[track_id].get('area', 1)),
        'age': 1,
        'consecutive_detections': tracker.active_tracks[track_id]['consecutive_detections'] + 1,
        'total_detections': tracker.active_tracks[track_id].get('total_detections', 1) + 1,
        'max_displacement': max(tracker.active_tracks[track_id].get('max_displacement', 0.0), displacement),
        'avg_intensity': avg_intensity,
    })
    tracker.active_tracks[track_id]['confidence'] = _calculate_confidence_for_tracker(
        tracker, tracker.active_tracks[track_id]
    )
    return tracker.active_tracks[track_id]


def _age_unmatched_track(tracker, track_id, decay=0.1):
    tracker.active_tracks[track_id]['age'] += 1
    tracker.active_tracks[track_id]['confidence'] = max(
        0.0,
        tracker.active_tracks[track_id].get('confidence', 0.5) - decay,
    )
    if track_id in tracker.kalman_filters:
        tracker.active_tracks[track_id]['kalman_centroid'] = tracker.kalman_filters[track_id].get_position()
    return tracker.active_tracks[track_id]


def _move_timed_out_tracks_to_lost(tracker):
    tracks_to_remove = []
    for track_id, track in list(tracker.active_tracks.items()):
        if track['age'] > tracker.track_timeout:
            tracks_to_remove.append(track_id)
            tracker.lost_tracks[track_id] = {
                'last_centroid': track.get('last_raw_centroid', track['centroid']),
                'velocity': tracker.kalman_filters[track_id].get_velocity() if track_id in tracker.kalman_filters else (0, 0),
                'lost_age': 0,
                'total_detections': track.get('total_detections', 1),
                'first_position': track.get('first_position'),
                'max_displacement': track.get('max_displacement', 0.0),
                'birth_frame': track.get('birth_frame', 0),
            }
    for track_id in tracks_to_remove:
        del tracker.active_tracks[track_id]
        if track_id in tracker.kalman_filters:
            del tracker.kalman_filters[track_id]
        if hasattr(tracker, 'track_particles') and track_id in tracker.track_particles:
            del tracker.track_particles[track_id]


def _complete_association_cycle(tracker, detections, track_ids, tracks, used_detections, matched_tracks, frame_idx, miss_decay=0.1):
    for track_id in track_ids:
        if track_id not in matched_tracks:
            tracks.append(_age_unmatched_track(tracker, track_id, decay=miss_decay))

    unmatched_dets = [detections[j] for j in range(len(detections)) if j not in used_detections]
    reidentified, reident_used = tracker.try_reidentify_lost_tracks(unmatched_dets)
    tracks.extend(reidentified)

    for j, det in enumerate(unmatched_dets):
        if j not in reident_used:
            new_track = tracker.create_new_track(det)
            new_track['birth_frame'] = frame_idx
            tracks.append(new_track)

    _move_timed_out_tracks_to_lost(tracker)
    return tracks


class IMMAdaptiveMotionTracker(AdvancedSatelliteTracker):
    """IMM-like tracker with mode-specific gating and motion constraints."""

    def __init__(self, *args, mode_profiles=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode_profiles = mode_profiles or {
            'search': dict(mahal=1.35, accel=1.8, direction=np.radians(150.0), speed=1.4),
            'cruise': dict(mahal=1.00, accel=1.0, direction=np.radians(100.0), speed=1.0),
            'maneuver': dict(mahal=1.40, accel=1.5, direction=np.radians(140.0), speed=1.2),
            'fast': dict(mahal=1.90, accel=2.2, direction=np.radians(175.0), speed=1.6),
        }

    def get_motion_mode(self, track_id):
        kf = self.kalman_filters.get(track_id)
        if kf is None:
            return 'search'

        speed = kf.get_speed()
        innovation = np.mean(kf.innovation_history) if len(kf.innovation_history) >= 3 else 0.0
        if speed > 8.0 or innovation > 12.0:
            return 'fast'
        if speed > 3.0 or innovation > 6.0:
            return 'maneuver'
        return 'cruise'

    def _profile_for(self, track_id, detection=None):
        profile = self.mode_profiles[self.get_motion_mode(track_id)].copy()
        area = detection.get('area', 1) if detection else 1
        size_scale = 1.0 + max(0.0, min(1.0, (area - 6) / 20.0))
        profile['accel'] *= size_scale
        profile['speed'] *= size_scale
        profile['mahal'] *= min(1.6, size_scale)
        return profile

    def check_motion_validity(self, track_id, new_detection):
        if track_id not in self.kalman_filters:
            return True

        kf = self.kalman_filters[track_id]
        track = self.active_tracks[track_id]
        profile = self._profile_for(track_id, new_detection)

        current_vel = kf.get_velocity()
        current_speed = kf.get_speed()
        last_pos = track.get('last_raw_centroid', track['centroid'])
        new_pos = new_detection['centroid']

        new_vel = (new_pos[0] - last_pos[0], new_pos[1] - last_pos[1])
        new_speed = np.sqrt(new_vel[0]**2 + new_vel[1]**2)

        if track['consecutive_detections'] < 3:
            step_cap = self.max_distance * 1.05 + 2.0
            if new_speed > step_cap:
                return False

        if new_speed > self.max_speed * profile['speed']:
            return False

        if track['consecutive_detections'] >= 3 and current_speed > self.min_speed:
            accel = np.sqrt((new_vel[0] - current_vel[0])**2 + (new_vel[1] - current_vel[1])**2)
            if accel > self.max_acceleration * profile['accel']:
                return False

        if track['consecutive_detections'] >= 3 and current_speed > 2.0 and new_speed > 2.0:
            dot_product = current_vel[0] * new_vel[0] + current_vel[1] * new_vel[1]
            cos_angle = dot_product / (current_speed * new_speed + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_diff = np.arccos(cos_angle)
            if angle_diff > max(self.max_direction_change, profile['direction']):
                return False

        return True

    def association_match_acceptable(self, track_id, detection, raw_cost):
        if raw_cost >= 1e8 - 1:
            return False
        if track_id not in self.kalman_filters:
            return raw_cost < self.max_distance
        kf = self.kalman_filters[track_id]
        profile = self._profile_for(track_id, detection)
        speed = kf.get_speed()
        thr = min(
            self.mahalanobis_threshold * profile['mahal'] * (1.0 + speed / 12.0),
            self.mahalanobis_threshold * 1.35,
        )
        det_pos = detection['centroid']
        pred = np.array(kf.get_position_float(), dtype=np.float64)
        dvec = np.array(det_pos, dtype=np.float64)
        pix = float(np.linalg.norm(dvec - pred))
        pix_cap = self.max_distance + self.velocity_gate_factor * speed * profile['speed'] + 6.0
        if raw_cost > thr or pix > pix_cap:
            return False
        return self.check_motion_validity(track_id, detection)

    def compute_association_costs(self, detections):
        if not self.active_tracks or not detections:
            return None, [], []

        track_ids = list(self.active_tracks.keys())
        cost_matrix = np.full((len(track_ids), len(detections)), 1e9)

        for i, track_id in enumerate(track_ids):
            kf = self.kalman_filters.get(track_id)
            track = self.active_tracks[track_id]

            if kf is not None:
                for j, det in enumerate(detections):
                    profile = self._profile_for(track_id, det)
                    mahal_dist = kf.get_mahalanobis_distance(det['centroid'])
                    speed = kf.get_speed()
                    adaptive_threshold = min(
                        self.mahalanobis_threshold * profile['mahal'] * (1.0 + speed / 12.0),
                        self.mahalanobis_threshold * 1.35,
                    )
                    pred = np.array(kf.get_position_float(), dtype=np.float64)
                    det_vec = np.array(det['centroid'], dtype=np.float64)
                    pixel_dist = float(np.linalg.norm(det_vec - pred))
                    pixel_cap = self.max_distance + self.velocity_gate_factor * speed * profile['speed'] + 6.0
                    if (
                        mahal_dist < adaptive_threshold
                        and pixel_dist <= pixel_cap
                        and self.check_motion_validity(track_id, det)
                    ):
                        cost_matrix[i, j] = mahal_dist
            else:
                last_pos = track.get('last_raw_centroid', track['centroid'])
                for j, det in enumerate(detections):
                    det_pos = det['centroid']
                    dist = np.sqrt((last_pos[0] - det_pos[0])**2 + (last_pos[1] - det_pos[1])**2)
                    if dist < self.max_distance:
                        cost_matrix[i, j] = dist

        return cost_matrix, track_ids, detections


class JPDALiteTracker(AdvancedSatelliteTracker):
    """Soft-assignment approximation of JPDA for cluttered frames."""

    def __init__(self, *args, jpda_temperature=0.65, jpda_margin=0.75, jpda_max_soft_neighbors=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.jpda_temperature = jpda_temperature
        self.jpda_margin = jpda_margin
        self.jpda_max_soft_neighbors = max(1, int(jpda_max_soft_neighbors))

    def _blend_detection(self, detections, candidate_indices, candidate_costs):
        costs = np.array(candidate_costs, dtype=np.float64)
        shifted = costs - costs.min()
        weights = np.exp(-shifted / max(self.jpda_temperature, 1e-6))
        weights /= weights.sum()

        primary = detections[candidate_indices[int(np.argmin(costs))]].copy()
        coords = np.array([detections[idx]['centroid'] for idx in candidate_indices], dtype=np.float64)
        centroid = tuple(np.round(np.sum(coords * weights[:, None], axis=0)).astype(int))
        primary['centroid'] = centroid
        primary['jpda_weights'] = weights.tolist()
        return primary, candidate_indices[int(np.argmin(costs))]

    def associate_tracks(self, detections, frame_idx):
        _predict_active_tracks(self)
        _refresh_lost_tracks(self)

        if not detections:
            for track in self.active_tracks.values():
                track['age'] += 1
                track['confidence'] = max(0.0, track.get('confidence', 0.5) - 0.1)
                if track['id'] in self.kalman_filters:
                    track['kalman_centroid'] = self.kalman_filters[track['id']].get_position()
            return list(self.active_tracks.values())

        if not self.active_tracks:
            reidentified, used_dets = self.try_reidentify_lost_tracks(detections)
            tracks = reidentified
            for j, det in enumerate(detections):
                if j not in used_dets:
                    new_track = self.create_new_track(det)
                    new_track['birth_frame'] = frame_idx
                    tracks.append(new_track)
            return tracks

        cost_matrix, track_ids, _ = self.compute_association_costs(detections)
        if cost_matrix is None:
            return []

        tracks = []
        used_detections = set()
        matched_tracks = set()

        row_order = sorted(range(len(track_ids)), key=lambda row: np.min(cost_matrix[row]))
        for row_idx in row_order:
            track_id = track_ids[row_idx]
            valid_cols = [
                col for col in range(len(detections))
                if col not in used_detections
                and self.association_match_acceptable(track_id, detections[col], cost_matrix[row_idx, col])
            ]
            if not valid_cols:
                continue

            costs = [cost_matrix[row_idx, col] for col in valid_cols]
            best_cost = min(costs)
            soft_cols = [
                col for col in valid_cols
                if cost_matrix[row_idx, col] <= best_cost + self.jpda_margin
            ][: self.jpda_max_soft_neighbors]
            soft_costs = [cost_matrix[row_idx, col] for col in soft_cols]
            blended_det, primary_idx = self._blend_detection(detections, soft_cols, soft_costs)

            track = _update_track_from_detection(self, track_id, blended_det)
            track['confidence'] = min(1.0, track['confidence'] + 0.03 * max(0, len(soft_cols) - 1))
            tracks.append(track)
            used_detections.add(primary_idx)
            matched_tracks.add(track_id)

        return _complete_association_cycle(
            self, detections, track_ids, tracks, used_detections, matched_tracks, frame_idx, miss_decay=0.08
        )


class MHTLiteTracker(AdvancedSatelliteTracker):
    """Delayed-commitment tracker inspired by MHT."""

    def __init__(self, *args, ambiguity_margin=0.42, **kwargs):
        super().__init__(*args, **kwargs)
        self.ambiguity_margin = ambiguity_margin

    def associate_tracks(self, detections, frame_idx):
        _predict_active_tracks(self)
        _refresh_lost_tracks(self)

        if not detections:
            for track in self.active_tracks.values():
                track['age'] += 1
                track['confidence'] = max(0.0, track.get('confidence', 0.5) - 0.08)
                if track['id'] in self.kalman_filters:
                    track['kalman_centroid'] = self.kalman_filters[track['id']].get_position()
            return list(self.active_tracks.values())

        if not self.active_tracks:
            reidentified, used_dets = self.try_reidentify_lost_tracks(detections)
            tracks = reidentified
            for j, det in enumerate(detections):
                if j not in used_dets:
                    new_track = self.create_new_track(det)
                    new_track['birth_frame'] = frame_idx
                    tracks.append(new_track)
            return tracks

        cost_matrix, track_ids, _ = self.compute_association_costs(detections)
        if cost_matrix is None:
            return []

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        tracks = []
        used_detections = set()
        matched_tracks = set()

        for row_idx, col_idx in zip(row_indices, col_indices):
            track_id = track_ids[row_idx]
            if self.association_match_acceptable(track_id, detections[col_idx], cost_matrix[row_idx, col_idx]):
                track = _update_track_from_detection(self, track_id, detections[col_idx])
                tracks.append(track)
                used_detections.add(col_idx)
                matched_tracks.add(track_id)

        for row_idx, track_id in enumerate(track_ids):
            if track_id in matched_tracks:
                continue

            remaining = [
                col for col in range(len(detections))
                if col not in used_detections
                and self.association_match_acceptable(track_id, detections[col], cost_matrix[row_idx, col])
            ]
            if not remaining:
                continue

            remaining.sort(key=lambda col: cost_matrix[row_idx, col])
            best_idx = remaining[0]
            second_idx = remaining[1] if len(remaining) > 1 else None

            if second_idx is not None:
                best_cost = cost_matrix[row_idx, best_idx]
                second_cost = cost_matrix[row_idx, second_idx]
                if second_cost - best_cost <= self.ambiguity_margin:
                    det_a = detections[best_idx]
                    det_b = detections[second_idx]
                    tentative = det_a.copy()
                    tentative['centroid'] = (
                        int(round(0.92 * det_a['centroid'][0] + 0.08 * det_b['centroid'][0])),
                        int(round(0.92 * det_a['centroid'][1] + 0.08 * det_b['centroid'][1])),
                    )
                    track = _update_track_from_detection(self, track_id, tentative)
                    track['confidence'] = max(0.25, track['confidence'] * 0.85)
                    track['hypothesis_state'] = 'tentative'
                    tracks.append(track)
                    used_detections.add(best_idx)
                    matched_tracks.add(track_id)

        return _complete_association_cycle(
            self, detections, track_ids, tracks, used_detections, matched_tracks, frame_idx, miss_decay=0.06
        )


class ParticleAssistedTracker(AdvancedSatelliteTracker):
    """Particle-assisted prediction for noisier motion and centroid jitter."""

    def __init__(self, *args, particle_count=64, particle_spread=2.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.particle_count = particle_count
        self.particle_spread = particle_spread
        self.track_particles = {}

    def _init_particles(self, track_id, position, velocity=(0.0, 0.0)):
        particles = np.tile(
            np.array([position[0], position[1], velocity[0], velocity[1]], dtype=np.float64),
            (self.particle_count, 1),
        )
        particles[:, :2] += np.random.normal(0, self.particle_spread, (self.particle_count, 2))
        particles[:, 2:] += np.random.normal(0, max(0.3, self.process_noise * 0.05), (self.particle_count, 2))
        self.track_particles[track_id] = particles

    def _predict_particles(self, track_id):
        if track_id not in self.track_particles:
            return None
        particles = self.track_particles[track_id]
        particles[:, :2] += particles[:, 2:]
        particles[:, :2] += np.random.normal(0, self.particle_spread, (self.particle_count, 2))
        particles[:, 2:] += np.random.normal(0, max(0.2, self.process_noise * 0.03), (self.particle_count, 2))
        self.track_particles[track_id] = particles
        return particles[:, :2].mean(axis=0)

    def _update_particles(self, track_id, measurement):
        if track_id not in self.track_particles:
            self._init_particles(track_id, measurement)
            return

        particles = self.track_particles[track_id]
        measurement = np.array(measurement, dtype=np.float64)
        dist2 = np.sum((particles[:, :2] - measurement) ** 2, axis=1)
        sigma2 = max(4.0, self.measurement_noise * 6.0)
        weights = np.exp(-dist2 / (2.0 * sigma2)) + 1e-12
        weights /= weights.sum()
        resampled_idx = np.random.choice(len(particles), size=len(particles), p=weights)
        particles = particles[resampled_idx]
        particles[:, :2] += np.random.normal(0, 0.75, (self.particle_count, 2))
        particles[:, 2:] = 0.7 * particles[:, 2:] + 0.3 * (measurement - particles[:, :2])
        self.track_particles[track_id] = particles

    def create_new_track(self, detection):
        track = super().create_new_track(detection)
        self._init_particles(track['id'], detection['centroid'])
        return track

    def try_reidentify_lost_tracks(self, detections):
        reidentified, used = super().try_reidentify_lost_tracks(detections)
        for track in reidentified:
            velocity = (0.0, 0.0)
            if track['id'] in self.kalman_filters:
                velocity = self.kalman_filters[track['id']].get_velocity()
            self._init_particles(track['id'], track['centroid'], velocity=velocity)
        return reidentified, used

    def association_match_acceptable(self, track_id, detection, raw_cost):
        """Blended costs need slightly looser Mahalanobis scale than pure Mahalanobis."""
        if raw_cost >= 1e8 - 1:
            return False
        if track_id not in self.kalman_filters:
            return raw_cost < self.max_distance
        kf = self.kalman_filters[track_id]
        speed = kf.get_speed()
        thr = min(
            self.mahalanobis_threshold * (1.0 + speed / 12.0),
            self.mahalanobis_threshold * 1.35,
        ) * 1.12
        det_pos = detection['centroid']
        pred = np.array(kf.get_position_float(), dtype=np.float64)
        dvec = np.array(det_pos, dtype=np.float64)
        pix = float(np.linalg.norm(dvec - pred))
        pix_cap = self.max_distance + self.velocity_gate_factor * speed + 6.0
        if raw_cost > thr or pix > pix_cap:
            return False
        return self.check_motion_validity(track_id, detection)

    def compute_association_costs(self, detections):
        if not self.active_tracks or not detections:
            return None, [], []

        track_ids = list(self.active_tracks.keys())
        cost_matrix = np.full((len(track_ids), len(detections)), 1e9)

        for i, track_id in enumerate(track_ids):
            track = self.active_tracks[track_id]
            kf = self.kalman_filters.get(track_id)
            particle_mean = self._predict_particles(track_id)

            if kf is not None:
                predicted = np.array(kf.get_position_float(), dtype=np.float64)
                if particle_mean is not None:
                    predicted = 0.5 * predicted + 0.5 * particle_mean
                speed = kf.get_speed()
                adaptive_threshold = min(
                    self.mahalanobis_threshold * (1.0 + speed / 12.0),
                    self.mahalanobis_threshold * 1.35,
                )
                kf_pred = np.array(kf.get_position_float(), dtype=np.float64)
                pixel_cap = self.max_distance + self.velocity_gate_factor * speed + 6.0
                for j, det in enumerate(detections):
                    det_pos = np.array(det['centroid'], dtype=np.float64)
                    mahal_dist = kf.get_mahalanobis_distance(det['centroid'])
                    particle_dist = np.linalg.norm(predicted - det_pos)
                    blended_cost = 0.7 * mahal_dist + 0.3 * (particle_dist / max(1.0, self.max_distance / 2.0))
                    pix_kf = float(np.linalg.norm(det_pos - kf_pred))
                    if (
                        blended_cost < adaptive_threshold
                        and pix_kf <= pixel_cap
                        and self.check_motion_validity(track_id, det)
                    ):
                        cost_matrix[i, j] = blended_cost
            else:
                last_pos = np.array(track.get('last_raw_centroid', track['centroid']), dtype=np.float64)
                for j, det in enumerate(detections):
                    det_pos = np.array(det['centroid'], dtype=np.float64)
                    dist = np.linalg.norm(last_pos - det_pos)
                    if dist < self.max_distance:
                        cost_matrix[i, j] = dist

        return cost_matrix, track_ids, detections

    def associate_tracks(self, detections, frame_idx):
        tracks = super().associate_tracks(detections, frame_idx)
        active_ids = set(self.active_tracks.keys())
        for track_id, track in self.active_tracks.items():
            self._update_particles(track_id, track.get('last_raw_centroid', track['centroid']))
        for track_id in list(self.track_particles.keys()):
            if track_id not in active_ids:
                del self.track_particles[track_id]
        return tracks


class TrackBeforeDetectTracker(AdvancedSatelliteTracker):
    """Temporal evidence accumulation before thresholding."""

    def __init__(self, *args, tbd_window=4, tbd_gain=0.75, **kwargs):
        super().__init__(*args, **kwargs)
        self.tbd_window = tbd_window
        self.tbd_gain = tbd_gain
        self.tbd_score_buffer = deque(maxlen=tbd_window)

    def preprocess_frame(self, frame):
        if self.use_temporal_smoothing and len(self.frame_buffer) == self.temporal_window:
            frame_smoothed = np.mean(list(self.frame_buffer), axis=0).astype(np.uint8)
        else:
            frame_smoothed = frame

        diff = cv2.absdiff(frame_smoothed, self.background).astype(np.float32)
        diff = cv2.GaussianBlur(diff, (3, 3), 0.3)
        self.tbd_score_buffer.append(diff)

        if len(self.tbd_score_buffer) > 1:
            stacked = np.stack(self.tbd_score_buffer, axis=0)
            accumulated = 0.6 * np.max(stacked, axis=0) + 0.4 * np.mean(stacked, axis=0)
        else:
            accumulated = diff

        threshold_value = max(1, int(round(self.bg_threshold * self.tbd_gain)))
        _, binary = cv2.threshold(accumulated.astype(np.uint8), threshold_value, 255, cv2.THRESH_BINARY)

        if self.use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        return binary


class AdaptiveRFSFamilyTracker(PoissonMultiBernoulliTracker):
    """PMBM / GLMB / LMB-inspired extension of the current PMB notebook code."""

    def detect_objects(self, binary_frame, original_frame):
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
                    'bbox': (x - 1, y - 1, 3, 3),
                    'area': 1,
                    'intensity': intensity,
                })
            return detections

        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples).fit(coords)
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:
                continue

            cluster_coords = [coords[i] for i in range(len(coords)) if clustering.labels_[i] == cluster_id]
            if not cluster_coords:
                continue

            area = len(cluster_coords)
            if area < self.min_detection_area or area > max(self.max_detection_area, 600):
                continue

            x_coords = [c[0] for c in cluster_coords]
            y_coords = [c[1] for c in cluster_coords]
            cx, cy = int(np.mean(x_coords)), int(np.mean(y_coords))
            intensities = [float(original_frame[y, x]) for x, y in cluster_coords]
            max_intensity = np.max(intensities)
            if max_intensity < self.min_intensity:
                continue

            detections.append({
                'centroid': (cx, cy),
                'bbox': (min(x_coords), min(y_coords), max(x_coords) - min(x_coords) + 1, max(y_coords) - min(y_coords) + 1),
                'area': area,
                'intensity': max_intensity,
            })

        return detections

    def _association_scale(self, component, detection):
        area = detection.get('area', 1)
        speed = component.get_speed()
        size_scale = 1.0 + min(1.4, max(0.0, (area - 8) / 25.0))
        speed_scale = 1.0 + min(0.8, speed / 20.0)
        return max(1.0, 0.65 * size_scale + 0.35 * speed_scale)

    def update_components(self, detections):
        n_components = len(self.bernoulli_components)
        n_detections = len(detections)

        if n_components == 0:
            for det in detections:
                pos = det['centroid']
                state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
                P = np.eye(4, dtype=np.float64) * 100.0
                component = BernoulliComponent(0.1, state, P, track_id=None)
                component.last_area = det.get('area', 1)
                component.last_bbox = det.get('bbox', (pos[0] - 1, pos[1] - 1, 3, 3))
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
                scale = self._association_scale(component, det)
                max_accel = self.max_acceleration * scale
                max_turn = min(170.0, self.max_direction_change + 20.0 * (scale - 1.0))
                max_speed = self.max_speed * scale
                if component.check_physical_constraints(det['centroid'], max_accel, max_turn, max_speed):
                    likelihood_matrix[i, j] = component.likelihood(det['centroid'])
                else:
                    valid_associations[i, j] = False
                    likelihood_matrix[i, j] = 1e-10

        clutter_intensity = self._clutter_intensity()
        updated_components = []
        used_detections = set()

        for i, component in enumerate(self.bernoulli_components):
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
                r_update = (self.detection_prob * component.r * likelihood) / (
                    self.detection_prob * component.r * likelihood + clutter_intensity * (1 - component.r) + 1e-10
                )
                if r_update > best_r and r_update > 0.1:
                    best_r = r_update
                    best_j = j

            if best_j >= 0:
                component_copy = deepcopy(component)
                component_copy.update(detections[best_j]['centroid'], detections[best_j]['intensity'])
                component_copy.r = best_r
                component_copy.last_area = detections[best_j].get('area', 1)
                component_copy.last_bbox = detections[best_j].get('bbox', None)
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
                component.last_area = det.get('area', 1)
                component.last_bbox = det.get('bbox', None)
                updated_components.append(component)

        self.bernoulli_components = updated_components

    def prune_and_merge(self):
        self.bernoulli_components = [c for c in self.bernoulli_components if c.r > self.pruning_threshold]
        for component in self.bernoulli_components:
            area = getattr(component, 'last_area', 1)
            speed = component.get_speed()
            dynamic_min_length = 3 if (speed > 4.0 or area > 12) else self.min_track_length
            if component.r > self.existence_threshold and component.track_id is None and component.detection_count >= dynamic_min_length:
                component.track_id = self.next_track_id
                self.next_track_id += 1
        self.bernoulli_components = [c for c in self.bernoulli_components if not (c.age > 50 and c.r < 0.3)]

    def get_confirmed_tracks(self):
        tracks = []
        for component in self.bernoulli_components:
            speed = component.get_speed()
            area = getattr(component, 'last_area', 1)
            dynamic_min_length = 3 if (speed > 4.0 or area > 12) else self.min_track_length
            dynamic_min_conf = max(0.45, self.min_display_confidence - (0.20 if speed > 4.0 or area > 12 else 0.0))

            if component.r > self.existence_threshold and component.track_id is not None and component.detection_count >= dynamic_min_length:
                confidence = component.calculate_confidence(self.min_confidence_frames, self.max_confidence_frames)
                if speed >= self.min_speed and speed <= self.max_speed * 1.5 and confidence >= dynamic_min_conf:
                    tracks.append({
                        'id': component.track_id,
                        'centroid': component.get_position(),
                        'existence_prob': component.r,
                        'confidence': confidence,
                        'speed': speed,
                        'age': component.age,
                        'detections': component.detection_count,
                        'area': area,
                    })
        return tracks


class PmbLargeFastTracker(AdaptiveRFSFamilyTracker):
    """
    PMB-family adaptive RFS tuned for larger / faster movers (**Current PMB** unchanged).

    Adds: softer fused foreground mask, supplemental percentile-based detections,
    stronger selective births, gentler pruning for large components, softer miss
    decay for established fast/large tracks, and short display hysteresis with
    centroid smoothing to reduce flicker.
    """

    def __init__(
        self,
        input_video_path,
        output_video_path,
        *,
        lf_bg_threshold_scale=0.72,
        lf_supplement_diff_percentile=88.0,
        lf_supplement_diff_floor=2.5,
        lf_supplement_intensity_relax=8,
        lf_supplement_min_area=4,
        lf_supplement_close_px=5,
        lf_supplement_merge_px=10.0,
        lf_strong_birth_area=18,
        lf_strong_birth_intensity=88.0,
        lf_strong_birth_r0=0.34,
        lf_moderate_birth_r0=0.22,
        lf_persist_speed_thr=5.0,
        lf_persist_area_thr=16,
        lf_miss_existence_floor_mult=0.94,
        lf_large_prune_area=36,
        lf_prune_scale_large=0.62,
        lf_display_miss_grace=5,
        lf_display_centroid_alpha=0.55,
        lf_display_confidence_decay=0.9,
        **kwargs,
    ):
        self.lf_bg_threshold_scale = float(lf_bg_threshold_scale)
        self.lf_supplement_diff_percentile = float(lf_supplement_diff_percentile)
        self.lf_supplement_diff_floor = float(lf_supplement_diff_floor)
        self.lf_supplement_intensity_relax = int(lf_supplement_intensity_relax)
        self.lf_supplement_min_area = int(lf_supplement_min_area)
        self.lf_supplement_close_px = int(lf_supplement_close_px)
        self.lf_supplement_merge_px = float(lf_supplement_merge_px)
        self.lf_strong_birth_area = int(lf_strong_birth_area)
        self.lf_strong_birth_intensity = float(lf_strong_birth_intensity)
        self.lf_strong_birth_r0 = float(lf_strong_birth_r0)
        self.lf_moderate_birth_r0 = float(lf_moderate_birth_r0)
        self.lf_persist_speed_thr = float(lf_persist_speed_thr)
        self.lf_persist_area_thr = int(lf_persist_area_thr)
        self.lf_miss_existence_floor_mult = float(lf_miss_existence_floor_mult)
        self.lf_large_prune_area = int(lf_large_prune_area)
        self.lf_prune_scale_large = float(lf_prune_scale_large)
        self.lf_display_miss_grace = int(lf_display_miss_grace)
        self.lf_display_centroid_alpha = float(lf_display_centroid_alpha)
        self.lf_display_confidence_decay = float(lf_display_confidence_decay)
        self._lf_display_cache: dict[int, dict] = {}
        super().__init__(input_video_path=input_video_path, output_video_path=output_video_path, **kwargs)

    def _association_scale(self, component, detection):
        base = super()._association_scale(component, detection)
        return min(base * 1.06, 2.25)

    def preprocess_frame(self, frame):
        if self.use_temporal_smoothing and len(self.frame_buffer) > 1:
            frame_smoothed = np.mean(list(self.frame_buffer), axis=0).astype(np.uint8)
        else:
            frame_smoothed = frame

        diff = cv2.absdiff(frame_smoothed, self.background)
        diff = cv2.GaussianBlur(diff, (3, 3), 0.3)
        _, hard = cv2.threshold(diff, self.bg_threshold, 255, cv2.THRESH_BINARY)
        soft_t = max(3, int(round(self.bg_threshold * self.lf_bg_threshold_scale)))
        _, soft = cv2.threshold(diff, soft_t, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_or(hard, soft)

        if self.use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        return binary

    def _lf_diff_map(self, gray_frame):
        if self.use_temporal_smoothing and len(self.frame_buffer) > 1:
            frame_smoothed = np.mean(list(self.frame_buffer), axis=0).astype(np.uint8)
        else:
            frame_smoothed = gray_frame
        diff = cv2.absdiff(frame_smoothed, self.background).astype(np.float32)
        diff = cv2.GaussianBlur(diff, (3, 3), 0.3)
        return diff

    def _lf_supplement_detections(self, gray_frame):
        diff = self._lf_diff_map(gray_frame)
        flat = diff.ravel()
        thr = max(
            self.lf_supplement_diff_floor,
            float(np.percentile(flat, self.lf_supplement_diff_percentile)),
        )
        mask = ((diff >= thr).astype(np.uint8)) * 255
        ksz = self.lf_supplement_close_px
        if ksz >= 3:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_a = max(self.min_detection_area, self.lf_supplement_min_area)
        max_a = max(int(self.max_detection_area), 600)
        relax_int = max(1, int(self.min_intensity) - self.lf_supplement_intensity_relax)
        out: list[dict] = []

        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < min_a or area > max_a:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw <= 0 or bh <= 0:
                continue
            roi = gray_frame[y : y + bh, x : x + bw]
            if roi.size == 0:
                continue
            max_i = float(np.max(roi))
            if max_i < relax_int:
                continue
            m = cv2.moments(c)
            if m["m00"] < 1e-6:
                continue
            cx = int(round(m["m10"] / m["m00"]))
            cy = int(round(m["m01"] / m["m00"]))
            out.append({
                "centroid": (cx, cy),
                "bbox": (x, y, bw, bh),
                "area": int(round(area)),
                "intensity": max_i,
            })
        return out

    def _lf_merge_detection_lists(self, primary: list[dict], extra: list[dict]) -> list[dict]:
        if not extra:
            return primary
        merged = list(primary)
        r2 = self.lf_supplement_merge_px ** 2
        for e in extra:
            cx, cy = e["centroid"]
            dup = False
            for p in merged:
                px, py = p["centroid"]
                if (cx - px) ** 2 + (cy - py) ** 2 <= r2:
                    dup = True
                    break
            if not dup:
                merged.append(e)
        return merged

    def _lf_birth_existence(self, det: dict) -> float:
        area = int(det.get("area", 1))
        inten = float(det.get("intensity", 0.0))
        if area >= self.lf_strong_birth_area and inten >= self.lf_strong_birth_intensity:
            return self.lf_strong_birth_r0
        if area >= 12 and inten >= float(self.min_intensity):
            return self.lf_moderate_birth_r0
        return 0.15

    def detect_objects(self, binary_frame, original_frame):
        base_dets = super().detect_objects(binary_frame, original_frame)
        extra = self._lf_supplement_detections(original_frame)
        return self._lf_merge_detection_lists(base_dets, extra)

    def update_components(self, detections):
        n_components = len(self.bernoulli_components)
        n_detections = len(detections)

        if n_components == 0:
            for det in detections:
                pos = det["centroid"]
                state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
                p = np.eye(4, dtype=np.float64) * 100.0
                r0 = self._lf_birth_existence(det)
                component = BernoulliComponent(r0, state, p, track_id=None)
                component.last_area = det.get("area", 1)
                component.last_bbox = det.get("bbox", (pos[0] - 1, pos[1] - 1, 3, 3))
                self.bernoulli_components.append(component)
            return

        if n_detections == 0:
            for component in self.bernoulli_components:
                r_miss = (1 - self.detection_prob) * component.r
                area = getattr(component, "last_area", 1)
                speed = component.get_speed()
                if component.track_id is not None and (
                    speed > self.lf_persist_speed_thr or area > self.lf_persist_area_thr
                ):
                    r_miss = max(r_miss, component.r * self.lf_miss_existence_floor_mult)
                component.r = r_miss
                component.consecutive_detections = 0
            return

        likelihood_matrix = np.zeros((n_components, n_detections))
        valid_associations = np.ones((n_components, n_detections), dtype=bool)

        for i, component in enumerate(self.bernoulli_components):
            for j, det in enumerate(detections):
                scale = self._association_scale(component, det)
                max_accel = self.max_acceleration * scale
                max_turn = min(170.0, self.max_direction_change + 20.0 * (scale - 1.0))
                max_speed = self.max_speed * scale
                if component.check_physical_constraints(det["centroid"], max_accel, max_turn, max_speed):
                    likelihood_matrix[i, j] = component.likelihood(det["centroid"])
                else:
                    valid_associations[i, j] = False
                    likelihood_matrix[i, j] = 1e-10

        clutter_intensity = self._clutter_intensity()
        updated_components = []
        used_detections = set()

        for i, component in enumerate(self.bernoulli_components):
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
                r_update = (self.detection_prob * component.r * likelihood) / (
                    self.detection_prob * component.r * likelihood
                    + clutter_intensity * (1 - component.r)
                    + 1e-10
                )
                if r_update > best_r and r_update > 0.1:
                    best_r = r_update
                    best_j = j

            if best_j >= 0:
                component_copy = deepcopy(component)
                component_copy.update(detections[best_j]["centroid"], detections[best_j]["intensity"])
                component_copy.r = best_r
                component_copy.last_area = detections[best_j].get("area", 1)
                component_copy.last_bbox = detections[best_j].get("bbox", None)
                updated_components.append(component_copy)
                used_detections.add(best_j)
            else:
                updated_components.append(best_component)

        for j, det in enumerate(detections):
            if j not in used_detections:
                pos = det["centroid"]
                state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
                p = np.eye(4, dtype=np.float64) * 100.0
                r0 = self._lf_birth_existence(det)
                component = BernoulliComponent(r0, state, p, track_id=None)
                component.update(pos, det["intensity"])
                component.last_area = det.get("area", 1)
                component.last_bbox = det.get("bbox", None)
                updated_components.append(component)

        self.bernoulli_components = updated_components

    def prune_and_merge(self):
        eff_thresh = float(self.pruning_threshold)
        pruned = []
        for c in self.bernoulli_components:
            la = getattr(c, "last_area", 1)
            t = eff_thresh * (self.lf_prune_scale_large if la > self.lf_large_prune_area else 1.0)
            if c.r > t:
                pruned.append(c)
        self.bernoulli_components = pruned

        for component in self.bernoulli_components:
            area = getattr(component, "last_area", 1)
            speed = component.get_speed()
            dynamic_min_length = 3 if (speed > 4.0 or area > 12) else self.min_track_length
            if (
                component.r > self.existence_threshold
                and component.track_id is None
                and component.detection_count >= dynamic_min_length
            ):
                component.track_id = self.next_track_id
                self.next_track_id += 1

        self.bernoulli_components = [
            c for c in self.bernoulli_components if not (c.age > 50 and c.r < 0.3)
        ]

    def get_confirmed_tracks(self):
        raw = super().get_confirmed_tracks()
        cache = self._lf_display_cache
        alpha = self.lf_display_centroid_alpha
        grace = max(0, self.lf_display_miss_grace)
        decay = self.lf_display_confidence_decay

        seen: set[int] = set()
        out: list[dict] = []

        for t in raw:
            tid = int(t["id"])
            seen.add(tid)
            cx, cy = t["centroid"]
            if tid in cache:
                ox, oy = cache[tid]["smooth"]
                sx = int(round(alpha * cx + (1.0 - alpha) * ox))
                sy = int(round(alpha * cy + (1.0 - alpha) * oy))
            else:
                sx, sy = cx, cy

            cache[tid] = {
                "smooth": (sx, sy),
                "miss": 0,
                "conf": float(t["confidence"]),
                "exist": float(t["existence_prob"]),
                "speed": float(t["speed"]),
                "age": int(t["age"]),
                "detections": int(t["detections"]),
                "area": int(t.get("area", 1)),
            }
            row = dict(t)
            row["centroid"] = (sx, sy)
            out.append(row)

        for tid in list(cache.keys()):
            if tid in seen:
                continue
            st = cache[tid]
            st["miss"] = int(st.get("miss", 0)) + 1
            if st["miss"] > grace:
                del cache[tid]
                continue

            conf = max(0.18, float(st.get("conf", 0.5)) * (decay ** st["miss"]))
            exist = max(0.05, float(st.get("exist", 0.4)) * (decay ** st["miss"]))
            sx, sy = st["smooth"]
            out.append({
                "id": tid,
                "centroid": (sx, sy),
                "existence_prob": exist,
                "confidence": conf,
                "speed": float(st.get("speed", 0.0)),
                "age": int(st.get("age", 0)),
                "detections": int(st.get("detections", 0)),
                "area": int(st.get("area", 1)),
            })

        return out


TRACKER_VARIANTS = [
    {
        'name': 'Advanced baseline',
        'family': 'Adaptive KF + Mahalanobis',
        'kind': 'advanced',
        'class': AdvancedSatelliteTracker,
        'output_tag': 'advanced_baseline',
        'overrides': {},
        'scores': dict(noise=3, motion=3, clutter=3, compute=2),
        'notes': 'Current adaptive-Kalman baseline.',
    },
    {
        'name': 'Current PMB',
        'family': 'PMB',
        'kind': 'pmb',
        'class': PoissonMultiBernoulliTracker,
        'output_tag': 'pmb_baseline',
        'overrides': {},
        'scores': dict(noise=3, motion=3, clutter=4, compute=3),
        'notes': 'Canonical PMB implementation (tracking_core/pmb.py).',
    },
    {
        'name': 'Textbook PMB',
        'family': 'PMB',
        'kind': 'pmb',
        'class': TextbookPoissonMultiBernoulliTracker,
        'output_tag': 'pmb_textbook',
        'overrides': dict(
            textbook_mahalanobis_gate_sq=9.21,
            min_display_confidence=0.48,
        ),
        'scores': dict(noise=3, motion=3, clutter=4, compute=4),
        'notes': 'Reference-style PMB: Mahalanobis gate, frame-scaled clutter, Hungarian assignment, Poisson-style birth r0; confidence = existence r.',
    },
    {
        'name': 'PMBM',
        'family': 'PMBM (multi-hypothesis)',
        'kind': 'pmb',
        'class': PoissonMultiBernoulliMixtureTracker,
        'output_tag': 'pmbm',
        'overrides': dict(
            textbook_mahalanobis_gate_sq=9.21,
            min_display_confidence=0.48,
            pmbm_k_best=5,
            pmbm_max_hypotheses=10,
        ),
        'scores': dict(noise=3, motion=3, clutter=4, compute=5),
        'notes': 'PMBM-style log-weighted mixture over association variants (Hungarian + forced-miss branches); MAP drives overlays and track log.',
    },
    {
        'name': 'PMB large/fast',
        'family': 'PMB',
        'kind': 'pmb',
        'class': PmbLargeFastTracker,
        'output_tag': 'pmb_large_fast',
        'overrides': dict(
            max_acceleration=24.0,
            max_direction_change=78.0,
            min_track_length=4,
            min_display_confidence=0.68,
            existence_threshold=0.48,
            pruning_threshold=0.008,
            max_detection_area=900,
            clutter_rate=6.0,
            detection_prob=0.82,
            lf_bg_threshold_scale=0.70,
            lf_supplement_diff_percentile=86.0,
            lf_supplement_intensity_relax=10,
            lf_display_miss_grace=6,
            lf_strong_birth_r0=0.36,
        ),
        'scores': dict(noise=3, motion=5, clutter=4, compute=3),
        'notes': 'Adaptive RFS path with relaxed gates for larger / faster movers; canonical Current PMB unchanged.',
    },
    {
        'name': 'PMB sparse (fast)',
        'family': 'PMB',
        'kind': 'pmb',
        'class': SparseBudgetPmTracker,
        'output_tag': 'pmb_sparse_fast',
        'overrides': dict(
            pmb_max_detections_per_frame=96,
            pmb_max_live_components=140,
            pmb_assoc_gate_pixels=50.0,
            pmb_min_birth_intensity=34.0,
            pmb_streaming_run=True,
            pmb_write_video_output=True,
            pmb_frame_scaled_clutter=True,
            min_intensity=30,
            pruning_threshold=0.014,
            existence_threshold=0.52,
            max_acceleration=14.0,
            max_speed=280.0,
        ),
        'scores': dict(noise=3, motion=4, clutter=4, compute=1),
        'notes': 'PMB-inspired greedy Bernoulli path with streaming I/O, spatial gate on likelihoods, detection/component caps, and frame-scaled clutter. Conservative reference remains Current PMB; permissive recall remains PMB large/fast.',
    },
    {
        'name': 'IMM adaptive',
        'family': 'IMM-like',
        'kind': 'advanced',
        'class': IMMAdaptiveMotionTracker,
        'output_tag': 'imm_adaptive',
        'overrides': dict(
            process_noise=10.0,
            max_speed=85.0,
            max_acceleration=45.0,
            mahalanobis_threshold=3.6,
            min_display_confidence=0.58,
            min_track_length=4,
            min_detection_confidence=0.22,
        ),
        'scores': dict(noise=4, motion=5, clutter=3, compute=3),
        'notes': 'Best first option for variable speed regimes.',
    },
    {
        'name': 'JPDA lite',
        'family': 'JPDA approximation',
        'kind': 'advanced',
        'class': JPDALiteTracker,
        'output_tag': 'jpda_lite',
        'overrides': dict(
            max_distance=22,
            max_acceleration=15.0,
            max_direction_change=20.0,
            max_speed=120.0,
            mahalanobis_threshold=3.2,
            min_display_confidence=0.80,
            min_track_length=5,
            bg_threshold=10,
            min_intensity=26,
            jpda_temperature=0.55,
            jpda_margin=0.45,
            jpda_max_soft_neighbors=1,
        ),
        'scores': dict(noise=4, motion=4, clutter=5, compute=4),
        'notes': 'Soft assignment for ambiguous detections.',
    },
    {
        'name': 'MHT lite',
        'family': 'MHT approximation',
        'kind': 'advanced',
        'class': MHTLiteTracker,
        'output_tag': 'mht_lite',
        'overrides': dict(
            track_timeout=26,
            lost_track_timeout=55,
            max_acceleration=15.0,
            max_direction_change=20.0,
            max_speed=120.0,
            min_display_confidence=0.80,
            min_track_length=5,
            bg_threshold=10,
            min_intensity=26,
            ambiguity_margin=0.38,
        ),
        'scores': dict(noise=4, motion=4, clutter=5, compute=5),
        'notes': 'Delayed commitment for ambiguous tracks.',
    },
    {
        'name': 'Particle assisted',
        'family': 'Particle filter hybrid',
        'kind': 'advanced',
        'class': ParticleAssistedTracker,
        'output_tag': 'particle_assisted',
        'overrides': dict(
            process_noise=10.0,
            measurement_noise=2.0,
            max_speed=120.0,
            max_acceleration=15.0,
            max_direction_change=20.0,
            min_display_confidence=0.80,
            min_track_length=5,
            mahalanobis_threshold=3.2,
            bg_threshold=10,
            min_intensity=26,
            particle_spread=1.5,
        ),
        'scores': dict(noise=5, motion=4, clutter=3, compute=5),
        'notes': 'Useful when centroids are noisy or streaky.',
    },
    {
        'name': 'Track-before-detect',
        'family': 'TBD (PMB soft evidence)',
        'kind': 'pmb',
        'class': TrueTbdPmTracker,
        'output_tag': 'track_before_detect',
        'overrides': dict(
            bg_threshold=10,
            min_intensity=26,
            max_acceleration=15.0,
            max_direction_change=20.0,
            max_speed=120.0,
            min_track_length=5,
            min_display_confidence=0.80,
            existence_threshold=0.55,
            clutter_rate=5.0,
            detection_prob=0.75,
            tbd_window=4,
            tbd_evidence_percentile=95.0,
            tbd_evidence_floor=2.5,
            tbd_peak_min_distance=10,
            tbd_soft_likelihood_gain=0.85,
            tbd_local_patch_radius=2,
            tbd_max_proposals_per_frame=8,
        ),
        'scores': dict(noise=5, motion=4, clutter=4, compute=4),
        'notes': 'True TBD-style: fused residual map (no global binary mask); PMB updates with soft local evidence.',
    },
    {
        'name': 'Temporal-accumulation TBD (approx)',
        'family': 'TBD (thresholded)',
        'kind': 'advanced',
        'class': TrackBeforeDetectTracker,
        'output_tag': 'tbd_temporal_approx',
        'overrides': dict(
            bg_threshold=8,
            min_intensity=18,
            max_distance=20,
            min_display_confidence=0.62,
            min_track_length=4,
            min_detection_confidence=0.24,
        ),
        'scores': dict(noise=5, motion=3, clutter=2, compute=4),
        'notes': 'Legacy baseline: temporal absdiff fusion then hard threshold + detect-then-track (not classical TBD).',
    },
    {
        'name': 'Adaptive RFS family',
        'family': 'PMBM / GLMB / LMB inspired',
        'kind': 'pmb',
        'class': AdaptiveRFSFamilyTracker,
        'output_tag': 'adaptive_rfs',
        'overrides': dict(
            max_acceleration=18.0,
            min_track_length=5,
            min_display_confidence=0.72,
            existence_threshold=0.52,
        ),
        'scores': dict(noise=4, motion=4, clutter=5, compute=4),
        'notes': 'Closest extension of the current PMB path.',
    },
]


def instantiate_tracker(
    spec: dict,
    input_video: str,
    tracker_output_dir: Path,
    *,
    preset_name: str | None = None,
    preset_file: str | Path | None = None,
    extra_kwargs: dict | None = None,
) -> object:
    """Build a tracker; writes to tracker_output_dir/output_video.mp4 and tracks.txt."""
    tracker_output_dir = Path(tracker_output_dir)
    tracker_output_dir.mkdir(parents=True, exist_ok=True)
    video_path = str(tracker_output_dir / "output_video.mp4")
    log_path = str(tracker_output_dir / "tracks.txt")
    base = ADVANCED_BASELINE_KWARGS if spec["kind"] == "advanced" else PMB_BASELINE_KWARGS
    kwargs = deepcopy(base)
    kwargs.update(spec.get("overrides", {}))
    kwargs, preset_meta = apply_preset_to_kwargs(
        kwargs,
        spec,
        preset_name=preset_name,
        preset_file=preset_file,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    kwargs["input_video_path"] = input_video
    kwargs["output_video_path"] = video_path
    kwargs["track_log_path"] = log_path
    tracker = spec["class"](**kwargs)
    tracker.preset_name = preset_meta["preset_name"]
    tracker.preset_file = preset_meta["preset_file"]
    return tracker


def compare_trackers(
    input_video: str,
    output_base: Path,
    run_all: bool = False,
    selected_names: set[str] | None = None,
) -> list[dict]:
    """Notebook-style comparison; each tracker gets a subfolder under output_base by output_tag."""
    rows: list[dict] = []
    selected_names = set(selected_names) if selected_names else None
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    for spec in TRACKER_VARIANTS:
        row = {
            "Algorithm": spec["name"],
            "Family": spec["family"],
            "Noise Robustness": spec["scores"]["noise"],
            "Motion Adaptability": spec["scores"]["motion"],
            "Clutter Handling": spec["scores"]["clutter"],
            "Compute Load": spec["scores"]["compute"],
            "Notes": spec["notes"],
            "Status": "not run",
            "Runtime (s)": float("nan"),
            "Tracks": float("nan"),
            "Detections": float("nan"),
            "Frames": float("nan"),
        }
        should_run = run_all and (selected_names is None or spec["name"] in selected_names)
        if should_run:
            try:
                sub = output_base / spec["output_tag"]
                tracker = instantiate_tracker(spec, input_video, sub)
                start_time = time.perf_counter()
                result = tracker.run()
                elapsed = time.perf_counter() - start_time
                row["Status"] = "ok"
                tr = result.get("runtime_s")
                row["Runtime (s)"] = (
                    round(float(tr), 2) if tr is not None else round(elapsed, 2)
                )
                row["Frames"] = result.get("frames_processed", float("nan"))
                row["Detections"] = result.get("total_detections", float("nan"))
                row["Tracks"] = result.get(
                    "unique_tracks", result.get("log_entries", float("nan"))
                )
            except Exception as exc:  # noqa: BLE001
                row["Status"] = f"error: {exc}"
        rows.append(row)
    return rows
