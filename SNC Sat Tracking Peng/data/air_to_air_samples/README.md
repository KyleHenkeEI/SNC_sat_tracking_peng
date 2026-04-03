# Air-to-air sample clips (optional)

Use this folder for **development and regression** clips when tuning the **`air_to_air`** preset and A2A preprocessing.

## Suggested layout

- Raw exports as received from the sensor (codec/container as appropriate).
- A short **`metadata.txt`** or sidecar JSON per clip: FPS, resolution, approximate host maneuver, and IR band if known.

## Git

Large media files are usually **not** committed. Add patterns to `.gitignore` if you store binaries here, or use your artifact store / share drive.

## Running trackers

From the project root:

```powershell
python run_pmb.py --input-video "data\air_to_air_samples\your_clip.mp4" --output-dir "out\a2a_test" --preset air_to_air --quiet
```

Compare with stabilization off by omitting the preset’s A2A flags: use `--preset default` and no `--a2a-phase-stabilize`, or duplicate the preset in a custom JSON with `"a2a_phase_stabilize": false`.

See [AIR_TO_AIR_ADAPTATION.md](../AIR_TO_AIR_ADAPTATION.md) for the full adaptation guide.
