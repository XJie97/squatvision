# SquatVision Project Status

Checkpoint: 20 October 2026

## Working core functionality

- MediaPipe-based pose estimation and hip/knee joint-angle analysis.
- Squat repetition counting and movement-phase detection.
- Minimum knee-angle frame identification.
- Bottom-pause estimation and video/JSON/CSV diagnostic exports.
- Manual surface annotation and review tools.
- Automated tests for core logic and experimental subsystems.

## Work in progress: IPF squat-depth detection

Fully automatic identification of the anatomical upper-thigh-at-hip
and knee-top reference points is not yet reliable.

Current experimental techniques include whole-person segmentation,
pose-guided boundary detection, RGB-gradient analysis, optical flow,
temporal consistency checks, and manual verification.

The reference squat recording has produced zero temporally accepted
automatic anatomical-surface pairs in the latest confirmed run.

SquatVision must report UNAVAILABLE where reliable anatomical
measurements cannot be established.

This checkpoint is considered a working development baseline for
pose analysis and repetition detection, NOT a validated automatic
IPF squat-depth judging system.
