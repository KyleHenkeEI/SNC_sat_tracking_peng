#!/usr/bin/env python3
"""Run selected tracker scripts via subprocess and aggregate comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE = Path(__file__).resolve().parent

# Tracker display name -> launcher script (repo root)
TRACKER_SCRIPT_BY_NAME: dict[str, str] = {
    "Advanced baseline": "run_advanced.py",
    "Current PMB": "run_pmb.py",
    "IMM adaptive": "run_imm.py",
    "JPDA lite": "run_jpda.py",
    "MHT lite": "run_mht.py",
    "Particle assisted": "run_particle.py",
    "Track-before-detect": "run_tbd.py",
    "Adaptive RFS family": "run_adaptive_rfs.py",
}


def _load_variants_registry() -> list[dict[str, Any]]:
    from tracking_core.variants import TRACKER_VARIANTS

    return TRACKER_VARIANTS


def parse_args() -> argparse.Namespace:
    default_out = WORKSPACE / "comparison_results"
    p = argparse.ArgumentParser(
        description="Run tracker CLIs and write comparison summaries + comparison videos.",
    )
    p.add_argument(
        "--input-video",
        type=Path,
        default=None,
        help="Input video path (overrides --input-dir if both are set).",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Folder of videos: uses first *.mp4 sorted by name (case-insensitive) if --input-video is omitted.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help="Root folder for this comparison run (creates a per-run subfolder).",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Subfolder name under --output-dir (default: timestamp).",
    )
    p.add_argument(
        "--trackers",
        nargs="*",
        default=None,
        help="Tracker names to run (default: all). Use --list-trackers for names.",
    )
    p.add_argument("--list-trackers", action="store_true", help="List tracker names and exit.")
    p.add_argument(
        "--skip-run",
        action="store_true",
        help="Only write qualitative summaries from the registry (no subprocess runs).",
    )
    p.add_argument(
        "--no-comparison-videos",
        action="store_true",
        help="Skip writing comparison_grid.mp4 and comparison_with_original.mp4.",
    )
    p.add_argument(
        "--capture-output",
        action="store_true",
        help="Buffer child process stdout/stderr (no live progress). Default streams to this terminal.",
    )
    return p.parse_args()


def resolve_input_video(
    input_video: Path | None,
    input_dir: Path | None,
) -> Path | None:
    """Return concrete video path from --input-video or first *.mp4 in --input-dir."""
    if input_video is not None:
        return input_video
    if input_dir is None:
        return None
    if not input_dir.is_dir():
        return None
    videos = sorted(
        input_dir.iterdir(),
        key=lambda p: p.name.lower(),
    )
    mp4s = [p for p in videos if p.is_file() and p.suffix.lower() == ".mp4"]
    if not mp4s:
        return None
    return mp4s[0]


def tracker_specs_by_name() -> dict[str, dict[str, Any]]:
    return {spec["name"]: spec for spec in _load_variants_registry()}


def select_specs(tracker_names: list[str] | None) -> list[dict[str, Any]]:
    specs = _load_variants_registry()
    if not tracker_names:
        return specs
    mapping = tracker_specs_by_name()
    missing = [n for n in tracker_names if n not in mapping]
    if missing:
        raise ValueError(
            "Unknown tracker names: "
            + ", ".join(missing)
            + ". Use --list-trackers to inspect valid names."
        )
    return [mapping[n] for n in tracker_names]


def save_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_rows_json(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def save_comparison_plot(rows: list[dict[str, Any]], output_path: Path) -> bool:
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    names = [row["Algorithm"] for row in rows]
    score_fields = [
        "Noise Robustness",
        "Motion Adaptability",
        "Clutter Handling",
        "Compute Load",
    ]
    score_matrix = np.array(
        [[row[field] for field in score_fields] for row in rows],
        dtype=np.float64,
    )

    has_runtime = any(row.get("Status") == "ok" for row in rows)
    fig_cols = 3 if has_runtime else 2
    fig, axes = plt.subplots(
        1,
        fig_cols,
        figsize=(7 * fig_cols, max(6, 0.45 * len(rows) + 4)),
    )
    if fig_cols == 1:
        axes = [axes]

    heatmap_ax = axes[0]
    im = heatmap_ax.imshow(score_matrix, cmap="viridis", aspect="auto", vmin=1, vmax=5)
    heatmap_ax.set_xticks(range(len(score_fields)))
    heatmap_ax.set_xticklabels(score_fields, rotation=20, ha="right")
    heatmap_ax.set_yticks(range(len(names)))
    heatmap_ax.set_yticklabels(names)
    heatmap_ax.set_title("Qualitative Score Matrix")
    for i in range(score_matrix.shape[0]):
        for j in range(score_matrix.shape[1]):
            heatmap_ax.text(
                j,
                i,
                int(score_matrix[i, j]),
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
    fig.colorbar(im, ax=heatmap_ax, fraction=0.046, pad=0.04)

    scatter_ax = axes[1]
    compute_values = [row["Compute Load"] for row in rows]
    motion_values = [row["Motion Adaptability"] for row in rows]
    noise_values = [row["Noise Robustness"] for row in rows]
    scatter = scatter_ax.scatter(
        compute_values,
        motion_values,
        s=np.array(noise_values) * 120,
        c=noise_values,
        cmap="plasma",
        alpha=0.85,
    )
    for x, y, name in zip(compute_values, motion_values, names):
        scatter_ax.text(x + 0.03, y + 0.03, name, fontsize=8)
    scatter_ax.set_xlabel("Compute Load (lower is cheaper)")
    scatter_ax.set_ylabel("Motion Adaptability")
    scatter_ax.set_title("Cost vs Motion Adaptability")
    scatter_ax.set_xlim(0.8, 5.4)
    scatter_ax.set_ylim(0.8, 5.4)
    scatter_ax.grid(alpha=0.25)
    fig.colorbar(
        scatter,
        ax=scatter_ax,
        fraction=0.046,
        pad=0.04,
        label="Noise Robustness",
    )

    if has_runtime:
        runtime_ax = axes[2]
        runtime_values = [
            0.0 if row.get("Runtime (s)") in (None, "", float("nan")) else float(row.get("Runtime (s)", 0.0))
            for row in rows
        ]
        runtime_ax.barh(names, runtime_values, color="steelblue")
        runtime_ax.set_title("Measured Runtime")
        runtime_ax.set_xlabel("Seconds")
        runtime_ax.invert_yaxis()
        runtime_ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Tracker Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def print_tracker_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'Algorithm':<22} {'Noise':>5} {'Motion':>6} "
        f"{'Clutter':>7} {'Load':>5} {'Status':<16} Notes"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['Algorithm']:<22} "
            f"{row['Noise Robustness']:>5} "
            f"{row['Motion Adaptability']:>6} "
            f"{row['Clutter Handling']:>7} "
            f"{row['Compute Load']:>5} "
            f"{str(row['Status']):<16} "
            f"{row['Notes']}"
        )


def read_summary(tracker_dir: Path) -> dict[str, Any] | None:
    p = tracker_dir / "summary.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def run_subprocess_tracker(
    script_name: str,
    input_video: Path,
    tracker_dir: Path,
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str | None]:
    script_path = WORKSPACE / script_name
    cmd = [
        sys.executable,
        str(script_path),
        "--input-video",
        str(input_video.resolve()),
        "--output-dir",
        str(tracker_dir.resolve()),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if capture_output:
        return subprocess.run(
            cmd,
            cwd=str(WORKSPACE),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    return subprocess.run(cmd, cwd=str(WORKSPACE), env=env)


def build_row_from_spec(spec: dict[str, Any], summary: dict[str, Any] | None, proc: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "Algorithm": spec["name"],
        "Family": spec["family"],
        "Noise Robustness": spec["scores"]["noise"],
        "Motion Adaptability": spec["scores"]["motion"],
        "Clutter Handling": spec["scores"]["clutter"],
        "Compute Load": spec["scores"]["compute"],
        "Notes": spec["notes"],
        "Status": "pending",
        "Runtime (s)": None,
        "Frames": None,
        "Detections": None,
        "Tracks": None,
        "Output Video": None,
        "Track Log": None,
    }
    if summary:
        st = summary.get("status")
        if st == "ok":
            row["Status"] = "ok"
            row["Runtime (s)"] = summary.get("runtime_s")
            row["Frames"] = summary.get("frames_processed")
            row["Detections"] = summary.get("total_detections")
            row["Tracks"] = summary.get("tracks_metric")
            row["Output Video"] = summary.get("output_video")
            row["Track Log"] = summary.get("track_log")
        elif st == "error":
            row["Status"] = f"error: {summary.get('error', 'unknown')}"
        else:
            row["Status"] = st or "unknown"
    elif proc is not None and proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
        row["Status"] = f"subprocess_exit_{proc.returncode}"
        row["Notes"] = (row["Notes"] + " | " + tail)[:500]
    return row


def write_manifest(
    run_dir: Path,
    input_video: str,
    rows: list[dict[str, Any]],
) -> None:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline": "terminal_subprocess",
        "workspace": str(WORKSPACE),
        "input_video": input_video,
        "summary_csv": str(run_dir / "comparison_summary.csv"),
        "summary_json": str(run_dir / "comparison_summary.json"),
        "plot_png": str(run_dir / "comparison_plot.png"),
        "comparison_grid": str(run_dir / "comparison_grid.mp4"),
        "comparison_with_original": str(run_dir / "comparison_with_original.mp4"),
        "trackers": rows,
    }
    with (run_dir / "comparison_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> int:
    args = parse_args()

    if args.list_trackers:
        print("Available trackers:")
        for spec in _load_variants_registry():
            print(f"- {spec['name']}: {spec['notes']}")
        return 0

    video_path = resolve_input_video(args.input_video, args.input_dir)
    if video_path is None:
        print(
            "Provide --input-video or --input-dir containing at least one .mp4 file.",
            file=sys.stderr,
        )
        return 1
    if not video_path.exists():
        print(f"Input video not found: {video_path}", file=sys.stderr)
        return 1
    if args.input_video is None and args.input_dir is not None:
        print(f"Using first video in folder (sorted by name): {video_path}")

    try:
        selected = select_specs(args.trackers)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir).resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    input_video = video_path.resolve()
    rows: list[dict[str, Any]] = []
    ok_videos: list[tuple[str, Path]] = []

    for spec in selected:
        name = spec["name"]
        tag = spec["output_tag"]
        tracker_dir = run_dir / tag
        script = TRACKER_SCRIPT_BY_NAME.get(name)
        if not script:
            rows.append(
                build_row_from_spec(
                    spec,
                    None,
                    None,
                )
            )
            rows[-1]["Status"] = "error: no launcher script mapped"
            continue

        summary = None
        proc = None
        if args.skip_run:
            row = build_row_from_spec(spec, None, None)
            row["Status"] = "not run"
            rows.append(row)
            continue

        tracker_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Running {name} ({script}) ===")
        t0 = time.perf_counter()
        proc = run_subprocess_tracker(
            script,
            input_video,
            tracker_dir,
            capture_output=args.capture_output,
        )
        wall = round(time.perf_counter() - t0, 2)
        if args.capture_output:
            if proc.stdout:
                print(proc.stdout, end="")
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr)

        summary = read_summary(tracker_dir)
        row = build_row_from_spec(spec, summary, proc)
        if row["Status"] == "pending":
            row["Status"] = f"error: no summary.json (exit {proc.returncode}, wall {wall}s)"
        rows.append(row)

        ov = row.get("Output Video")
        if ov and Path(ov).is_file() and row.get("Status") == "ok":
            ok_videos.append((name, Path(ov)))

    print()
    print_tracker_table(rows)

    save_rows_csv(rows, run_dir / "comparison_summary.csv")
    save_rows_json(rows, run_dir / "comparison_summary.json")
    plot_saved = save_comparison_plot(rows, run_dir / "comparison_plot.png")
    write_manifest(run_dir, str(input_video), rows)

    if not args.skip_run and not args.no_comparison_videos and len(ok_videos) >= 2:
        from tracking_core.comparison_video import (
            write_comparison_grid,
            write_comparison_with_original,
        )

        paths = [p for _, p in ok_videos]
        labels = [n for n, _ in ok_videos]
        grid_ok = write_comparison_grid(paths, labels, run_dir / "comparison_grid.mp4")
        montage_ok = write_comparison_with_original(
            input_video,
            paths,
            labels,
            run_dir / "comparison_with_original.mp4",
        )
    else:
        grid_ok = False
        montage_ok = False

    print("\nSaved outputs:")
    print(f"- Run directory: {run_dir}")
    print(f"- Summary CSV: {run_dir / 'comparison_summary.csv'}")
    print(f"- Summary JSON: {run_dir / 'comparison_summary.json'}")
    print(f"- Manifest: {run_dir / 'comparison_manifest.json'}")
    if plot_saved:
        print(f"- Comparison plot: {run_dir / 'comparison_plot.png'}")
    else:
        print("- Comparison plot: skipped (matplotlib unavailable)")
    if grid_ok:
        print(f"- Comparison grid video: {run_dir / 'comparison_grid.mp4'}")
    if montage_ok:
        print(f"- Comparison + original: {run_dir / 'comparison_with_original.mp4'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
