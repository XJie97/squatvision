# SquatVision changelog

## 3.0.0 — 2026-10-18

- Added optional manual hip-crease and knee-top reference point calibration.
- Added per-frame thigh-attached propagation of both depth points in `squatvision/depth.py`.
- Added independent depth readout and dashed guides to annotated MP4 and all detected phase screenshots.
- Added signed vertical depth and classification in `events.csv`, `events.json`, `frames.csv` and attempt summaries.
- Added calibration preview and regression/integration tests.
- Preserved legacy angle tracking and explicitly marked uncalibrated anatomical depth as unavailable.

# Changelog

## 2.0.0 — 2026-10-16

- Rebuilt repetition segmentation to require stable standing calibration, substantial flexion, hip movement and stable return to upright.
- Incomplete attempts no longer count as completed repetitions.
- Added temporal event markers and PNG screenshots for the beginning/end of descent, start of ascent, return to upright and suspected failure.
- Added separate visual dashed rays with gaps around hip and knee centres, including angle arcs and measurement labels.
- Added CSV/JSON event output, tunable conservative thresholds and clip time selection.
- Added multi-scenario unit tests for rep-count regressions, pauses, reversals and missing landmarks.
- Left true source-video accuracy unverified until an annotated ground-truth example is available.
