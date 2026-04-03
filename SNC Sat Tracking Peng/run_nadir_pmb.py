#!/usr/bin/env python3
"""
Run the Nadir PMB tracker on a LEO/MEO downward-looking video.

Usage:
    python run_nadir_pmb.py                          # defaults to synthetic_LEO_NIR.mp4
    python run_nadir_pmb.py --input my_video.mp4
    python run_nadir_pmb.py --method affine
    python run_nadir_pmb.py --snr 4.0                # more sensitive detection
    python run_nadir_pmb.py --no-video
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nadir_tracking.nadir_pmb import NadirPMBTracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nadir PMB tracker for space-looking-down video")
    p.add_argument("--input", "-i", type=str,
                   default=str(ROOT / "synthetic_LEO_NIR.mp4"),
                   help="Input video path")
    p.add_argument("--output-dir", "-o", type=str,
                   default=str(ROOT / "_nadir_output"),
                   help="Output directory")
    p.add_argument("--method", "-m", type=str, default="phase",
                   choices=["homography", "affine", "phase"],
                   help="Ego-motion registration method (phase is immune to target contamination)")
    p.add_argument("--stack-depth", type=int, default=1,
                   help="Number of frames for residual accumulation")
    p.add_argument("--no-video", action="store_true",
                   help="Disable output video writing")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)

    # Detection
    p.add_argument("--snr", type=float, default=4.0,
                   help="SNR threshold for peak detection (lower = more sensitive)")
    p.add_argument("--peak-min-residual", type=int, default=8,
                   help="Minimum absolute residual value for a peak")
    p.add_argument("--peak-min-distance", type=int, default=12,
                   help="Minimum pixel distance between detection peaks")

    # Coasting
    p.add_argument("--coast-frames", type=int, default=20,
                   help="How many missed frames before dropping an established track")
    p.add_argument("--coast-floor", type=float, default=0.55,
                   help="Minimum existence probability while coasting (must be > existence-threshold)")

    # PMB tuning
    p.add_argument("--existence-threshold", type=float, default=0.45)
    p.add_argument("--min-display-confidence", type=float, default=0.45)
    p.add_argument("--clutter-rate", type=float, default=3.0)
    p.add_argument("--detection-prob", type=float, default=0.80)
    p.add_argument("--birth-rate", type=float, default=0.08)
    p.add_argument("--process-noise", type=float, default=4.0)
    p.add_argument("--max-speed", type=float, default=15.0)
    p.add_argument("--max-acceleration", type=float, default=15.0)
    p.add_argument("--max-direction-change", type=float, default=75.0)
    p.add_argument("--min-speed", type=float, default=3.0)
    p.add_argument("--min-track-length", type=int, default=4)
    p.add_argument("--cluster-eps", type=int, default=5)
    p.add_argument("--max-detection-area", type=int, default=500)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video = str(out_dir / "nadir_tracked.mp4")
    track_log = str(out_dir / "nadir_tracks.txt")

    print(f"\n  Input video:  {args.input}")
    print(f"  Output dir:   {out_dir}")
    print(f"  Method:       {args.method}")
    print(f"  SNR thresh:   {args.snr}")
    print(f"  Coast frames: {args.coast_frames}\n")

    tracker = NadirPMBTracker(
        input_video_path=args.input,
        output_video_path=output_video,
        # ego-motion
        registration_method=args.method,
        orb_max_features=1000,
        residual_stack_depth=args.stack_depth,
        # detection
        snr_threshold=args.snr,
        peak_min_residual=args.peak_min_residual,
        peak_min_distance=args.peak_min_distance,
        # coasting
        coast_frames=args.coast_frames,
        coast_existence_floor=args.coast_floor,
        # PMB parameters
        existence_threshold=args.existence_threshold,
        min_display_confidence=args.min_display_confidence,
        clutter_rate=args.clutter_rate,
        detection_prob=args.detection_prob,
        birth_rate=args.birth_rate,
        process_noise_q=args.process_noise,
        max_speed=args.max_speed,
        max_acceleration=args.max_acceleration,
        max_direction_change=args.max_direction_change,
        min_speed=args.min_speed,
        min_track_length=args.min_track_length,
        cluster_eps=args.cluster_eps,
        max_detection_area=args.max_detection_area,
        # output
        write_video_output=not args.no_video,
        track_log_path=track_log,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        verbose=True,
    )

    result = tracker.run()

    print("\n  Results summary:")
    for k, v in result.items():
        print(f"    {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
