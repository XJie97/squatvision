# Real-video verification and limitations — v5

## Inputs inspected

The original user-owned `squat.mp4` was available in the execution environment.
Metadata observed: **352×640**, **29.990 FPS**, **2862 frames**. The squat is
approximately within **70–76 seconds**; the v4 report identifies one complete
repetition, bottom around **72.758 seconds**.

Frames from 71.49, 72.30, 72.75, 73.00, 73.40 and 73.60 seconds were reviewed.
The visible shorts obscure the exact thigh-at-hip skin surface in some frames.
A pose-derived or automatically followed marker must not be treated as ground truth.

## What was actually executed

- `python -m unittest discover -s tests -v`: **41 tests passed** (26 original + 15 new).
- Ran independent `manual_depth.py --no-gui` against the **actual original video**
  using the previously supplied *calibration selection* at ~71.490s:
  upper-thigh=(128,274), top-of-knee=(202,329) in original pixels.
- The manual-only tool generated valid JSON, CSV and a screenshot with the
  **arithmetic margin = -55 px**, as expected from these particular selections.

**This is not the maximum depth.** The tested frame predates the bottom; the
calibration selections are not verified ground-truth anatomical landmarks, and
this arithmetic check cannot determine whether the lift reached IPF depth.

## What was not verified

- Full MediaPipe inference on the original 64–80-second clip could not run in
  the execution environment because `mediapipe` was unavailable.
- No independently validated sequence of actual anatomical surface-point
  labels was provided for the bottom; quantitative mean/max localisation
  errors, tracking availability with v5, and threshold-agreement statistics
  therefore **cannot yet be reported**.
- All v5 flow-based improvements are structural (extra independent verified
  manual anchors) rather than a proven increase in accuracy on this clip.

## Required remaining video validation

1. Install dependencies and run the existing v4 pose pipeline on the 64–80s window.
2. Manually label several frames across approximately 72–74 seconds in
   `manual_depth.py`, including the deepest **relative surface** position.
3. Compare those independently selected coordinates against the automatic
   tracker with `validate_depth.py`, reporting mean/max error, tracking
   availability and margin error. Do not relabel failed automatic frames as
   ground truth.
4. Inspect `depth_vs_time.svg` for long gaps and implausible jumps and use
   manual measurements where automated tracking remains invalid.
5. Compare classification only on reliable reviewed frames. Without the full
   anatomical surface being visible, an uncertain result is appropriate.

## v5.2 review of user-supplied `annotated(3).mp4`

The uploaded 16-second clip covers approximately 64–80 seconds of the original
squat. Inspection at source-relative 6.0, 7.0, 8.0, 8.75, 9.3, 10 and 11
seconds shows that optical-flow depth is intermittent. A numerical margin
around 72 seconds is followed by **UNAVAILABLE** near the reported bottom
around 72.758 seconds. Consequently, the earlier **EST. ABOVE** label was
not defensible as a conclusion about the entire repetition.

v5.2 was tested by automated regression tests, not by rerunning its complete
MediaPipe analysis on this video. The MediaPipe package and model are absent
from this environment. The underlying anatomy is partly concealed by clothing,
and no independently reviewed bottom-position surface labels were supplied.
No numeric IPF depth margin for the full squat, localisation error or
competition outcome has been validated.
