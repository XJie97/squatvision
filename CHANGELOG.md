# SquatVision changelog

## v5 — Experimental IPF surface-depth and manual fallback

- Added `squatvision/ipf_depth.py` for independent surface-height comparison,
  signed/normalised margin, explicit measurement provenance, engineering uncertainty.
- Added `manual_depth.py` for source-frame manual annotations, editing,
  zoom controls, CSV/JSON, screenshots, independent of MediaPipe or optical flow.
- Added `squatvision/ipf_plot.py` with missing-tracking gaps and manual points.
- Integrated annotations and manual-anchor recovery into `app.py`.
- Updated portrait-video depth overlays and fixed-position knee-height reference line.
- Added 15 regression and end-to-end tests, preserving v4's 26 original tests.
- Preserved phase screenshots, repetition detection and legacy output columns.

**Not an official IPF depth determination.**
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

## v4 — Independent local visual tracking and depth diagnostics

- Replace pose-only reference-point propagation with independent LK optical-flow visual tracking initiated from actual manually selected source-video pixels; reject inadequate texture and inconsistent motion.
- Add manual multi-anchor calibration to support validated reinitialisation after visual tracking loss.
- Add zoomable marker-editor, pose reference overlays, pixel offsets, customizable point selection and nearby-frame visual preview.
- Add uncertainty-aware 2D depth status, normalisation by projected thigh length, peak-depth frame and timestamp, and diagnostic SVG plots.
- Add `validate_depth.py` for independent annotated-frame reference error measurement.
- Preserve squat repetition state machine and event screenshots.
- The actual recorded squat has not been provided to this environment: no real-video accuracy claim is made.

### Real-video inspection follow-up

- Located and examined original `squat.mp4`, confirming significant horizontal body movement and temporary obstruction.
- Detected an additional visual optical-flow failure on the actual footage: high-confidence background attachment near the knee and low-texture loss near the hip reference.
- Added a stronger pose-motion gate and multiple manually clicked reference points for reinitialisation, as well as a short-clip sequential grayscale cache.
- MediaPipe and its model are unavailable in this sandbox; see `REAL_VIDEO_VALIDATION.md` for exactly what was and was not verified.

## v5.2 — 2026-10-18
- Prevent provisional optical-flow measurements from being labelled EST. ABOVE
  or EST. BELOW in normal video reports.
- Do not infer whole-repetition insufficient depth from an early measurement
  when tracking fails near the bottom.
- Make manually annotated below-threshold frames separately identifiable.
- Compact 352px portrait overlays; avoid full-width knee lines and on-leg
  angle text; maintain angle guides and event timestamps.
- Mark diagnostic optical-flow graph traces explicitly UNVERIFIED.
- Add tracking-loss and sparse-manual-coverage regression tests.
