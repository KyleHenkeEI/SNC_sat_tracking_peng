#!/usr/bin/env python3
"""
Overnight parameter sweep: generate ensemble configs, run sequentially,
produce summary.csv for morning review.

Usage:
    python sweep_ensembles.py --input-video <path_to_video.mp4> [--hours 7] [--workers 4]

Output:
    _sweep_output/sweep_YYYYMMDD_HHMMSS/
        summary.csv                         <-- open first
        001_pmb_trio_dirchg15/
            consensus_overlay.mp4           <-- review these
            ensemble_manifest.json
        002_adv_quartet_dirchg20/
            ...
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from copy import deepcopy
from itertools import product
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Base tracker "recipes" — split by tracker family so PMB-only params
# never reach AdvancedSatelliteTracker (which rejects them).
# ---------------------------------------------------------------------------

# Parameters accepted by BOTH families
SHARED_CLEAR = dict(
    min_display_confidence=0.80, min_track_length=5,
    max_speed=120, max_acceleration=15, max_direction_change=20,
    bg_threshold=10, min_intensity=26,
)
SHARED_FAST = dict(
    min_display_confidence=0.75, min_track_length=5,
    min_speed=12.0,
    max_speed=350, max_acceleration=25, max_direction_change=4,
    bg_threshold=10, min_intensity=26,
)

# PMB-family only (PoissonMultiBernoulliTracker and subclasses)
PMB_CLEAR_EXTRA = dict(
    existence_threshold=0.55, clutter_rate=5.0, detection_prob=0.75,
)
PMB_FAST_EXTRA = dict(
    existence_threshold=0.50, clutter_rate=5.0, detection_prob=0.75,
    process_noise_q=50.0,
)

# Advanced-family only (AdvancedSatelliteTracker and subclasses)
ADV_CLEAR_EXTRA = dict(
    process_noise=10.0, measurement_noise=2.0, mahalanobis_threshold=3.2,
    max_distance=22,
)
ADV_FAST_EXTRA = dict(
    process_noise=12.0, measurement_noise=2.0, mahalanobis_threshold=3.6,
    max_distance=30,
)

# Which tracker names belong to which family
PMB_FAMILY_NAMES = {
    "Current PMB", "Textbook PMB", "PMB sparse (fast)",
    "Track-before-detect", "Adaptive RFS family",
}
ADV_FAMILY_NAMES = {
    "Particle assisted", "JPDA lite", "MHT lite",
    "IMM adaptive", "Advanced baseline",
}

# Keys that only PMB-family accepts (strip from Advanced overrides)
PMB_ONLY_KEYS = {
    "existence_threshold", "clutter_rate", "detection_prob", "birth_rate",
    "survival_prob", "pruning_threshold", "process_noise_q", "min_association_r",
    "tbd_evidence_percentile", "tbd_evidence_floor", "tbd_peak_min_distance",
    "tbd_soft_likelihood_gain", "tbd_local_patch_radius",
    "tbd_max_proposals_per_frame", "tbd_window",
    "textbook_mahalanobis_gate_sq",
    "pmb_max_detections_per_frame", "pmb_max_live_components",
    "pmb_assoc_gate_pixels", "pmb_min_birth_intensity",
    "pmb_streaming_run", "pmb_write_video_output", "pmb_frame_scaled_clutter",
}

# Keys that only Advanced-family accepts (strip from PMB overrides)
ADV_ONLY_KEYS = {
    "process_noise", "measurement_noise", "mahalanobis_threshold",
    "max_distance", "track_timeout", "lost_track_timeout",
    "velocity_gate_factor", "min_detection_confidence",
    "particle_spread", "particle_count",
    "jpda_temperature", "jpda_margin", "jpda_max_soft_neighbors",
    "ambiguity_margin",
}

PMB_TRIO = [
    ("pmb",      "Current PMB"),
    ("txtpmb",   "Textbook PMB"),
    ("sparse",   "PMB sparse (fast)"),
]
ADV_QUARTET = [
    ("tbd",      "Track-before-detect"),
    ("particle", "Particle assisted"),
    ("jpda",     "JPDA lite"),
    ("mht",      "MHT lite"),
]
MIXED_BEST = [
    ("pmb",      "Current PMB"),
    ("txtpmb",   "Textbook PMB"),
    ("tbd",      "Track-before-detect"),
    ("particle", "Particle assisted"),
]

FUSION_DEFAULT = dict(
    quorum=2, spatial_threshold_px=6.0, link_threshold_px=12.0,
    pool_overrides=dict(fast=dict(spatial_threshold_px=15.0, link_threshold_px=50.0)),
)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _build_overrides(tracker_name: str, shared: dict, user_overrides: dict,
                     pmb_extra: dict, adv_extra: dict) -> dict:
    """Build a clean override dict for one tracker, filtering out params
    that would crash the wrong tracker family."""
    is_pmb = tracker_name in PMB_FAMILY_NAMES
    is_adv = tracker_name in ADV_FAMILY_NAMES

    ov = {**shared}
    if is_pmb:
        ov.update(pmb_extra)
    elif is_adv:
        ov.update(adv_extra)

    # Merge user overrides, but strip keys the family doesn't accept
    for k, v in user_overrides.items():
        if is_pmb and k in ADV_ONLY_KEYS:
            continue
        if is_adv and k in PMB_ONLY_KEYS:
            continue
        ov[k] = v

    # Per-tracker defaults
    if tracker_name == "Track-before-detect":
        ov.setdefault("tbd_evidence_percentile", 95.0)
        ov.setdefault("tbd_evidence_floor", 2.5)
        ov.setdefault("tbd_peak_min_distance", 10)
        ov.setdefault("tbd_soft_likelihood_gain", 0.85)
        ov.setdefault("tbd_max_proposals_per_frame", 8)
    elif tracker_name == "Particle assisted":
        ov.setdefault("particle_spread", 1.5)
    elif tracker_name == "Textbook PMB":
        ov.setdefault("textbook_mahalanobis_gate_sq", 9.21)
    elif tracker_name == "PMB sparse (fast)":
        ov.setdefault("pmb_max_detections_per_frame", 96)
    elif tracker_name == "MHT lite":
        ov.setdefault("track_timeout", 26)
        ov.setdefault("lost_track_timeout", 55)
    elif tracker_name == "Current PMB":
        ov.setdefault("birth_rate", 0.06)

    return ov


def make_ensemble_config(
    trackers: list[tuple[str, str]],
    clear_overrides: dict,
    fast_overrides: dict,
    fusion: dict,
    description: str,
) -> dict:
    runs = []
    for key_pfx, tracker_name in trackers:
        clear_ov = _build_overrides(
            tracker_name, SHARED_CLEAR, clear_overrides,
            PMB_CLEAR_EXTRA, ADV_CLEAR_EXTRA,
        )
        fast_ov = _build_overrides(
            tracker_name, SHARED_FAST, fast_overrides,
            PMB_FAST_EXTRA, ADV_FAST_EXTRA,
        )

        # Fast-pool TBD gets slightly looser evidence settings
        if tracker_name == "Track-before-detect":
            fast_ov.setdefault("tbd_evidence_percentile", 93.0)
            fast_ov.setdefault("tbd_evidence_floor", 2.0)
            fast_ov.setdefault("tbd_peak_min_distance", 8)
            fast_ov.setdefault("tbd_soft_likelihood_gain", 0.90)
            fast_ov.setdefault("tbd_max_proposals_per_frame", 10)
        if tracker_name == "Textbook PMB":
            fast_ov.setdefault("textbook_mahalanobis_gate_sq", 16.0)
        if tracker_name == "PMB sparse (fast)":
            fast_ov.setdefault("pmb_max_detections_per_frame", 64)
        if tracker_name == "Current PMB":
            fast_ov.setdefault("min_association_r", 0.01)

        runs.append(dict(
            key=f"{key_pfx}_clear", tracker_name=tracker_name,
            label=f"{key_pfx} clear", pool="clear", overrides=clear_ov,
        ))
        runs.append(dict(
            key=f"{key_pfx}_fast", tracker_name=tracker_name,
            label=f"{key_pfx} fast", pool="fast", overrides=fast_ov,
        ))
    return dict(description=description, fusion=fusion, runs=runs)


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

def generate_experiments() -> list[tuple[str, dict]]:
    """Sweep bg_threshold x min_intensity x clutter_rate with PMB trio.

    Quorum = 2 (2/3 trackers must agree).
    Combos sorted center-out so the most promising runs happen first.
    """
    experiments: list[tuple[str, dict]] = []
    idx = 0

    fusion_q2 = deepcopy(FUSION_DEFAULT)
    fusion_q2["quorum"] = 2

    def add(name_suffix, trackers, clear_ov, fast_ov=None, fusion=None):
        nonlocal idx
        idx += 1
        tag = f"{idx:03d}_{name_suffix}"
        fov = fast_ov or {}
        fus = fusion or deepcopy(fusion_q2)
        cfg = make_ensemble_config(trackers, clear_ov, fov, fus, tag)
        experiments.append((tag, cfg))

    bg_values  = [6, 8, 10, 12, 14]
    mi_values  = [20, 22, 24, 26, 28, 30]
    cr_values  = [2.0, 4.0, 6.0, 8.0, 12.0]

    combos = list(product(bg_values, mi_values, cr_values))
    combos.sort(key=lambda t: (
        ((t[0] - 10) / 4) ** 2 +
        ((t[1] - 26) / 5) ** 2 +
        ((t[2] - 6.0) / 5) ** 2
    ))

    for bg, mi, cr in combos:
        ov = dict(bg_threshold=bg, min_intensity=mi, clutter_rate=cr)
        add(f"bg{bg}_mi{mi}_cr{cr}", PMB_TRIO, ov, ov, deepcopy(fusion_q2))

    return experiments


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one_experiment(
    tag: str,
    config: dict,
    input_video: Path,
    sweep_dir: Path,
    workers: int,
) -> dict:
    out_dir = sweep_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "ensemble_config.json"
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    cmd = [
        sys.executable, str(WORKSPACE / "run_ensemble.py"),
        "--input-video", str(input_video),
        "--ensemble-config", str(cfg_path),
        "--output-dir", str(out_dir),
        "--workers", str(workers),
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(WORKSPACE))
    wall_s = time.perf_counter() - t0

    manifest_path = out_dir / "ensemble_manifest.json"
    row: dict = {
        "tag": tag,
        "wall_s": round(wall_s, 1),
        "status": "FAIL",
    }

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        row["consensus_tracks"] = manifest.get("consensus_tracks_total", "")
        for pool_name, pool_info in manifest.get("pools", {}).items():
            row[f"pool_{pool_name}_tracks"] = pool_info.get("consensus_tracks", "")

        tracker_results = manifest.get("tracker_results", [])
        n_total = len(tracker_results)
        n_ok = sum(1 for r in tracker_results if r.get("status") == "ok")
        if n_ok == n_total and result.returncode == 0:
            row["status"] = "ok"
        elif n_ok > 0:
            row["status"] = f"partial ({n_ok}/{n_total})"
        else:
            row["status"] = "FAIL"
    else:
        row["consensus_tracks"] = ""
        if result.returncode != 0:
            err_tail = (result.stderr or "")[-500:]
            row["error"] = err_tail

    row["description"] = config.get("description", "")

    if config.get("runs"):
        first_ov = config["runs"][0].get("overrides", {})
        for k in [
            "bg_threshold", "min_intensity", "existence_threshold",
            "min_display_confidence", "clutter_rate", "detection_prob",
            "max_direction_change", "max_acceleration",
        ]:
            row[k] = first_ov.get(k, "")
    fus = config.get("fusion", {})
    row["quorum"] = fus.get("quorum", "")
    row["spatial_px"] = fus.get("spatial_threshold_px", "")
    return row


CSV_FIELDS = [
    "tag", "wall_s", "status", "consensus_tracks",
    "pool_clear_tracks", "pool_fast_tracks",
    "bg_threshold", "min_intensity", "existence_threshold",
    "min_display_confidence", "clutter_rate", "detection_prob",
    "max_direction_change", "max_acceleration",
    "quorum", "spatial_px", "description",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Overnight ensemble parameter sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Morning review:\n"
            "  1. Open _sweep_output/sweep_*/summary.csv\n"
            "  2. Sort by consensus_tracks to find the sweet spot\n"
            "  3. Watch consensus_overlay.mp4 in promising folders\n"
            "  4. Use --start-at N to resume after interruption\n"
        ),
    )
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=7.0,
                        help="Time budget in hours (default: 7)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers per ensemble run (default: 4)")
    parser.add_argument("--start-at", type=int, default=1,
                        help="Resume from experiment N (skip earlier ones)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Sweep output directory (default: auto-timestamped)")
    args = parser.parse_args()

    input_video = args.input_video.resolve()
    if not input_video.is_file():
        print(f"Video not found: {input_video}", file=sys.stderr)
        return 1

    if args.output_dir is not None:
        sweep_dir = args.output_dir.resolve()
    else:
        sweep_dir = WORKSPACE / "_sweep_output" / time.strftime("sweep_%Y%m%d_%H%M%S")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    experiments = generate_experiments()
    deadline = time.time() + args.hours * 3600
    csv_path = sweep_dir / "summary.csv"

    print("=" * 72)
    print(f"  ENSEMBLE PARAMETER SWEEP")
    print(f"  Experiments : {len(experiments)}")
    print(f"  Time budget : {args.hours}h")
    print(f"  Workers     : {args.workers}")
    print(f"  Video       : {input_video}")
    print(f"  Output      : {sweep_dir}")
    if args.start_at > 1:
        print(f"  Resuming at : experiment {args.start_at}")
    print("=" * 72, flush=True)

    csv_mode = "a" if args.start_at > 1 and csv_path.is_file() else "w"
    with open(csv_path, csv_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if csv_mode == "w":
            writer.writeheader()

        completed = 0
        for i, (tag, config) in enumerate(experiments, 1):
            if i < args.start_at:
                continue

            remaining_s = deadline - time.time()
            if remaining_s < 180:
                print(
                    f"\n*** Time budget exhausted ({args.hours}h). "
                    f"Completed {completed} experiments "
                    f"(up to #{i - 1} of {len(experiments)}). ***"
                )
                print(f"*** Resume with: --start-at {i} --output-dir \"{sweep_dir}\" ***")
                break

            remaining_min = remaining_s / 60.0
            print(
                f"\n[{i}/{len(experiments)}] {tag}  "
                f"({remaining_min:.0f} min remaining)",
                flush=True,
            )

            row = run_one_experiment(tag, config, input_video, sweep_dir, args.workers)
            writer.writerow(row)
            f.flush()
            completed += 1

            tracks = row.get("consensus_tracks", "?")
            print(
                f"   -> {row['status']}, "
                f"{tracks} consensus tracks, "
                f"{row['wall_s']}s wall time",
                flush=True,
            )

    print(f"\nSweep complete. {completed} experiments finished.")
    print(f"  Summary CSV       : {csv_path}")
    print(f"  Consensus videos  : {sweep_dir}/*/consensus_overlay.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
