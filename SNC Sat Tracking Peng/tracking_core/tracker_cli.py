"""Shared CLI entry for single-tracker terminal runs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from tracking_core.presets import preset_descriptions
from tracking_core.variants import TRACKER_VARIANTS, instantiate_tracker


def _spec_by_name(name: str) -> dict[str, Any]:
    for spec in TRACKER_VARIANTS:
        if spec["name"] == name:
            return spec
    raise KeyError(name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one satellite tracker variant.")
    p.add_argument("--input-video", type=Path, required=False)
    p.add_argument("--output-dir", type=Path, required=False, help="Folder for output_video.mp4, tracks.txt, summary.json")
    p.add_argument("--quiet", action="store_true", help="Less console output from the tracker")
    p.add_argument("--preset", type=str, default=None, help="Preset name from tracker_presets.json")
    p.add_argument("--preset-file", type=Path, default=None, help="Optional custom preset JSON file")
    p.add_argument("--list-presets", action="store_true", help="List available preset names and exit")
    return p.parse_args()


def main_for(tracker_name: str) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    if args.list_presets:
        print("Available presets:")
        for name, desc in preset_descriptions(args.preset_file).items():
            print(f"- {name}: {desc}")
        return 0
    try:
        spec = _spec_by_name(tracker_name)
    except KeyError:
        print(f"Unknown tracker name: {tracker_name}", file=sys.stderr)
        return 2

    if args.input_video is None or args.output_dir is None:
        print("--input-video and --output-dir are required unless --list-presets is used.", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_video = str(args.input_video.resolve())

    summary: dict[str, Any] = {
        "tracker_name": tracker_name,
        "output_tag": spec["output_tag"],
        "family": spec["family"],
        "input_video": input_video,
        "output_dir": str(args.output_dir.resolve()),
        "preset_name": args.preset,
        "preset_file": str(args.preset_file.resolve()) if args.preset_file else None,
        "status": "pending",
        "runtime_s": None,
        "frames_processed": None,
        "total_detections": None,
        "tracks_metric": None,
        "output_video": None,
        "track_log": None,
        "error": None,
    }

    try:
        tracker = instantiate_tracker(
            spec,
            input_video,
            args.output_dir,
            preset_name=args.preset,
            preset_file=args.preset_file,
        )
        if args.quiet:
            tracker.verbose = False
        summary["output_video"] = str(Path(tracker.output_video_path).resolve())
        summary["track_log"] = str(Path(tracker.track_log_path).resolve())

        t0 = time.perf_counter()
        result = tracker.run()
        summary["runtime_s"] = round(time.perf_counter() - t0, 2)
        summary["status"] = "ok"
        summary["frames_processed"] = result.get("frames_processed")
        summary["total_detections"] = result.get("total_detections")
        summary["tracks_metric"] = result.get("unique_tracks", result.get("log_entries"))
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "error"
        summary["error"] = str(exc)
        out_json = args.output_dir / "summary.json"
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out_json = args.output_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Done: {tracker_name} in {summary['runtime_s']}s -> {summary['output_video']}")
    return 0
