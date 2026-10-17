# SquatVision v4: real-footage inspection status

## Source inspected

- Original: `squat.mp4` (user-supplied Library recording, **not redistributed**).
- Resolution: 352 × 640 (portrait); approximately 29.99 FPS, 2,862 frames.
- Analysis target: approximately 70–76 seconds; previous run found one complete squat.
- Previous calibration frame: ~71.49 s (frame 2144), right side.
- Previous manual point selections: hip crease (128,274), knee top (202,329), **in original-video pixels**.

## Confirmed by inspection and isolated optical-flow experiment

- The athlete's hip and knee positions move substantially during descent and ascent; an anatomical marker stationary in image X is wrong.
- A first version of the *visual-only* LK tracker was not sufficient. The hip-reference feature group lost inliers when tracking away from the calibration frame. The knee-reference feature group remained nearly fixed at (202,330) through frames where the visible knee had moved: the flow had attached to background or an unrelated textured region. High LK consistency is **not** proof of anatomical correctness.
- The local flow implementation was therefore changed to compare each independently tracked point with a **coarse joint-motion expectation** from the existing MediaPipe pose stream, using a relatively tight pose-consistency gate. The pose does **not** supply replacement anatomical point coordinates. Inconsistent flow must become unavailable. Separately labelled keyframes support legitimate manual recovery after tracking loss.
- Short clips use a sequential grayscale frame cache rather than repeated video seeking.

## Not yet verified

This environment can decode the original MP4 with OpenCV, but does not have MediaPipe installed and cannot access the package/model distribution over the network. Therefore the fully integrated MediaPipe + optical-flow results, measured pixel errors versus manually labelled anatomical landmarks, the final one-rep count under v4 and the depth estimate **have not been verified on the actual footage here**.

No competition-depth verdict has been calculated. The validated synthetic regressions test two-dimensional X/Y motion, missing data, multiple manual anchors and unchanged event exports; they are not evidence of actual anatomical precision.

## How to finish real validation on the user's computer

1. Recalibrate the original source at 71.49s in v4. The old v3 JSON lacks exact clicked point coordinates.
2. Run the 64–80s segment. Inspect screenshots and `diagnostics/*.svg`, particularly X displacement and confidence gaps.
3. Label at least 5–10 frames (standing, early descent, bottom, ascent, lockout) **from the original MP4**, not an already annotated render.
4. Compare actual manually clicked X/Y references against `frames.csv` using `validate_depth.py` and report errors and coverage. If optical flow loses the correct anatomical feature, add a new manual calibration keyframe and repeat.
5. Only treat the output as an **informational 2D depth estimate**, regardless of a positive signed depth difference.
