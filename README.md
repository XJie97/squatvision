# SquatVision v5 — IPF Surface-Depth Assessment (Experimental)

**This is the v4-to-v5 extension.** Existing MediaPipe joint-centre angle measurements,
repetition state machine, movement-phase screenshots, optical-flow tracking and
other v4 features remain supported.

**Not an IPF-certified judging system.** The IPF squat rule compares the *top of the
thigh at the hip joint* against the *top of the knee*. Neither point is equivalent
to MediaPipe's hip/knee joint centre; the knee or hip angle alone does not determine
competition depth. Camera perspective, shorts, lighting, occlusions, surface point
ambiguity and foreshortening remain limitations.

## What's new in v5

- Independent **IPF anatomical surface-depth** calculations (never replace these with joint centres).
- Signed image-plane margin: `hip_surface_y_px - knee_top_y_px` (positive is below).
- Projected-thigh-normalised ratio, **only when valid MediaPipe joint landmarks** are
  available for the projected hip-to-knee segment length. Neither metric is centimetres.
- Explicit measurement provenance: `AUTOMATED_TRACKING`, `MANUAL_ANNOTATION`,
  `MANUAL_VERIFIED_TRACKING`, or `NONE` when unmeasurable.
- Explicit four-way classifications: `ESTIMATED_BELOW_DEPTH_THRESHOLD`,
  `ESTIMATED_ABOVE_DEPTH_THRESHOLD`, `BORDERLINE_UNCERTAIN`, `UNAVAILABLE`.
- User-configured positive error bound for manual point placement (default ±6 px),
  and conservative heuristic error bound for automated tracking (8 px baseline plus
  a tracking-quality penalty). **These are assumed engineering tolerances, not calibrated
  confidence intervals.** Near-zero margins must be treated as uncertain.
- Manual frame-by-frame fallback even with no reliable optical-flow tracking;
  annotating multiple frames near the bottom is recommended. Manual frames may
  serve as verified additional optical-flow anchors when a valid pose exists.
- Depth-vs-time SVG, explicit gaps for unavailable automatic observations, manual
  annotation markers and engineering error bars.
- A new depth status readout on annotated portrait videos, knee-height reference
  guide, annotated peak-depth and manual frame screenshots.
- Extended outputs: `ipf_depth_assessments` in `summary.json`, independent margin and
  uncertainty fields in `frames.csv`, and manual depth results in JSON and CSV.

## Quick start on Windows PowerShell

From the extracted repository root, using **Python 3.11/3.12**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_model.py
python -m unittest discover -s tests -v
```

If you already have v4 installed, you only need to update the source files and
rerun the tests. Video, model and calibration files are not bundled.

### A. Manually assess the actual squat bottom (works without MediaPipe)

```powershell
python manual_depth.py --input data/squat.mp4 --output data/manual_depth.json --start-time 72 --end-time 74 --uncertainty-px 8
```

Interactive controls:

| Action | Control |
|---|---|
| Previous/next frame | A / D |
| Move 10 frames | J / L |
| Select anatomical references | Left-click the top of thigh at hip joint, then top of knee |
| Correct current frame | Click again to replace its two selected points |
| Clear label for current frame | R or right mouse button |
| Zoom | + / - |
| Save annotations and screenshots | S |
| Exit review | Esc (current annotations are still saved) |

Review several frames in the bottom region rather than treating the minimum
knee-angle frame as maximum anatomical depth. The user may not be able to identify
the exact skin boundary under clothing. In that case increase the stated marking
uncertainty or mark the point unavailable; do not invent a precise reference.

You can also insert a manual annotation without a GUI (use actual source-frame pixel
coordinates; the following values are **placeholders**, not judging data):

```powershell
python manual_depth.py --input data/squat.mp4 --output data/manual_depth.json --no-gui --point FRAME,HIP_X,HIP_Y,KNEE_X,KNEE_Y
```

Manual-only exports next to the JSON file:
`manual_depth_results.csv`, `manual_depth_results.json`,
`manual_depth_vs_time.svg`, and `manual_depth_screenshots/*.png`.
A normalised margin is intentionally unavailable here because no joint-centre
thigh-length reference is inferred. The signed pixel margin still works.

### B. Combine manual labels with the existing v4 analysis

```powershell
python app.py --input data/squat.mp4 --output outputs/v5_manual --side right --start-time 64 --end-time 80 --min-visibility 0.30 --manual-annotations data/manual_depth.json
```

This works even when all optical-flow observations are missing. The existing
MediaPipe model remains required for the main app's **repetition detection**, but
not for `manual_depth.py`'s independent anatomical-depth assessment.

### C. Automated flow plus manual anchors (recommended when usable)

```powershell
python app.py --input data/squat.mp4 --output outputs/v5_combined --side right --start-time 64 --end-time 80 --min-visibility 0.30 --depth-calibration depth_calibration_v4.json --manual-annotations data/manual_depth.json
```

You can also use `--depth-calibration depth_calibration_v4.json` without manual
labels, but in the reference recording this can report `UNAVAILABLE` around the
bottom. The application **does not fabricate a valid measurement** in that case.

## IPF depth math and classification

Image Y increases downward. At an annotated or reliably tracked frame:

```text
margin_px = upper_thigh_at_hip_y_px - knee_top_y_px
normalised_margin = margin_px / projected_hip_joint_to_knee_joint_length_px
```

The projected length uses pose joint centres **only as a scale reference**, never
as a substitute for the surface heights. If joint pose is missing, normalised
margin is blank, but manual signed pixel depth is still available.

- Margin greater than +assumed uncertainty: `ESTIMATED_BELOW_DEPTH_THRESHOLD`.
- Margin less than -assumed uncertainty: `ESTIMATED_ABOVE_DEPTH_THRESHOLD`.
- Margin within the ±assumed uncertainty band: `BORDERLINE_UNCERTAIN`.
- Missing or unreliable reference points: `UNAVAILABLE`.

For each repetition, the maximum is computed from the **highest observed signed
margin**, independent of the minimum knee angle. Tracking gaps are not interpolated.
If the result is 'above', this describes the **best available observed frame**;
unobserved frames may conceal a deeper position. It cannot prove a competition
failure. Similarly 'below' is an estimated 2D result, not an official ruling.

## Main outputs

```text
outputs/v5_combined/
├── annotated.mp4
├── frames.csv
├── attempts.csv
├── events.csv
├── events.json
├── summary.json                 # includes ipf_depth_assessments
├── depth_vs_time.svg            # with missing data shown as gaps
├── event_screenshots/          # all existing movement phases
├── depth_screenshots/          # manually labelled / maximum-depth frames
└── diagnostics/                # original v4 trajectory plots
```

New per-frame fields include `upper_thigh_at_hip_x/y`,
`knee_top_surface_x_px/y_px`, `depth_margin_px`, `depth_margin_normalized`,
`ipf_depth_status`, `depth_uncertainty_px`, `measurement_source`,
`tracking_quality`, and `unavailable_reason`. Existing v4 `knee_top_x/y` fields
retain their **normalised** coordinates for compatibility. Original legacy
`depth_status`/`crease_depth_status` columns are retained in lowercase; use the
new `ipf_depth_status` for the four explicit assessment classifications.

## How to validate accuracy against manually labelled frames

Run `validate_depth.py` with manually verified reference annotations. Report
mean/max localisation error, depth-margin error, valid tracking fraction, and
threshold agreement on **independent** labels. An accepted calibration does not
prove tracked anatomical accuracy. See `REAL_VIDEO_VALIDATION.md` for the reference
clip's observed limitations.

## Privacy and scope

- Keep original gym videos private unless everyone shown has permitted disclosure.
- The `.gitignore` excludes `data/`, `models/`, `outputs/`, video formats, and `depth_calibration*.json`; inspect `git status` before publishing calibration JSON.
- Do not use these estimates for official IPF judging, medical decisions or
  injury-risk claims.

---

## Original v4 documentation

# SquatVision

**Video-based squat analysis with independent hip-crease-to-knee-top depth tracking, using Python, MediaPipe and OpenCV.**

SquatVision is a computer vision prototype that analyses **recorded, side-view squat videos** to estimate joint angles, identify movement phases, count completed repetitions and export timestamped screenshots. It is designed as a practical demonstration of human pose estimation, temporal event detection and reproducible video analysis.

> **Important:** SquatVision is an experimental analysis tool, **not** a certified powerlifting judge, clinical assessment tool or substitute for a qualified coach. Its depth, lockout and failure labels are approximate, camera-dependent measurements.

## Features

- **Human pose estimation:** Uses the pretrained MediaPipe Pose Landmarker to estimate shoulder, hip, knee and ankle positions.
- **Joint-angle analysis:** Measures projected 2D knee and hip angles, with dashed visual guides showing the angle geometry.
- **Calibrated anatomical depth estimate (v3):** Select visible top hip crease and top knee surface on a reference frame; track estimated positions throughout the video independently of joint-angle analysis.
- **Phase-specific depth evidence:** Annotates crease/knee markers, dashed vertical-height guides and a signed difference on every detected movement-phase screenshot, with values in CSV/JSON.
- **Repetition detection:** Uses smoothed knee angles, visible hip displacement and sustained upright postures to distinguish completed reps from partial movements.
- **Movement-phase timestamps:** Marks the start and end of descent, start of ascent and estimated return to lockout.
- **Bottom-pause estimation:** Estimates the interval between the detected end of descent and beginning of ascent.
- **Incomplete and suspected failed attempts:** Records unfinished attempts separately; flags only certain observed stalls/reversals as **suspected** failures.
- **Video and data exports:** Creates an annotated MP4, event screenshots, CSV files and JSON summaries.
- **Configurable thresholds:** Supports adjustment of repetition sensitivity, body-side selection and analysis time windows.

## How it works

```text
Recorded video (MP4)
        |
        v
OpenCV: decode video frames
        |
        v
MediaPipe: estimate body landmarks
        |
        v
Geometry: calculate 2D joint angles and hip position
        |
        +--> Optional calibrated HC / KT point propagation
        |
        v
Temporal analysis: smooth observations and segment squat attempts
        |
        v
Output: annotated video + event screenshots + CSV/JSON reports
```

The rep detector does **not** count every isolated knee bend. It looks for a movement that begins from a sufficiently stable upright posture, contains meaningful knee flexion **and** hip displacement, and returns to a sustained upright position. Attempts without that completed return are recorded separately rather than counted as completed repetitions.

## Requirements

- Python **3.11 or 3.12** recommended
- Windows, macOS or Linux
- A short, constant-frame-rate MP4 recording
- Internet access **once** to download the pretrained pose model
- CPU execution is supported; no dedicated GPU is required

Main dependencies: `mediapipe`, `opencv-python`, `numpy`.

## Installation

Clone your repository or open the extracted project folder:

```bash
git clone https://github.com/YOUR_USERNAME/squatvision.git
cd squatvision
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**

```bat
.venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install packages and download the MediaPipe pose model:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python download_model.py
```

If PowerShell prevents virtual-environment activation, commands can be run directly with `.venv\Scripts\python.exe` instead of `python` (for example, `.venv\Scripts\python.exe app.py --help`).

## Usage

### 1. Prepare the video

Create a `data` folder and place a recording at `data/squat.mp4`. For more dependable measurements:

- Record **one person**, primarily from the side, with the entire body visible.
- Keep the camera stationary, preferably on a tripod.
- Avoid strong occlusion, abrupt camera movement and large changes in lighting.
- Use footage that you own or have permission to analyse.

### 2. Analyse the squat

```bash
python app.py --input data/squat.mp4 --output outputs/demo
```

The script prints an analysis summary and exports annotated video, event screenshots and structured data into `outputs/demo/`.

### 3. Calibrate hip-crease-to-knee-top depth tracking (new)

**Important:** MediaPipe provides hip/knee *joint centres*, not the anatomical **top of hip crease** or **top of knee joint surface**. These cannot be inferred with competition-grade precision from the 33 default pose landmarks. SquatVision therefore supports optional **manual two-point calibration**, and propagates those points using the visible thigh geometry. It does **not** claim to visually rediscover skin folds or the kneecap in every frame.

First, find a representative, unobstructed **side-view frame** (ideally near the lowest squat position) in which both anatomical reference points are visible. Select its actual timestamp in seconds:

```powershell
python calibrate_depth.py --input data/squat.mp4 --time 39.2 --side right --output depth_calibration.json
```

The `39.2` timestamp is illustrative. Click **(1) the top of the hip crease**, then **(2) the top surface of the knee joint**. Press **Enter** to save; **R** to reset; **Esc** to cancel. `--side right` means the athlete's right side, not the right half of the screen. You can use `--side auto` if the visible side is unclear. The GUI automatically scales large frames so they fit the screen.

Check `outputs/depth_calibration_preview.png` and confirm that HC (hip crease) and KT (knee top) are located appropriately. After calibration:

```powershell
python app.py --input data/squat.mp4 --output outputs/demo --depth-calibration depth_calibration.json
```

All critical event frames—**descent start, descent end, ascent start, estimated lockout and suspected failure (when detected)**—will show both reference-point estimates, dashed horizontal guides, the vertical separation indicator and the depth readout, alongside the existing knee and hip angle guides. Each event will also receive a row in `events.csv` with its signed vertical depth and classification.

**Without calibration**, the project continues to report knee/hip angles, repetition phases and the existing hip-joint depth proxy; anatomical depth is explicitly labelled **NOT CALIBRATED**. This prevents a hip-joint centre from being mislabeled as the actual hip crease.

For machines without an OpenCV display, both manually determined reference positions can be supplied as `X,Y` coordinates in pixels of the **original full-resolution frame**:

```powershell
python calibrate_depth.py --input data/squat.mp4 --time 39.2 --side right --hip-crease 142,352 --knee-top 167,363 --output depth_calibration.json
```

These coordinates are examples, not suitable for arbitrary video. The tool still uses MediaPipe on the chosen frame to establish the thigh reference. The generated `depth_calibration.json` is ignored by Git and should be regenerated when camera position, resolution, subject or tracked side changes.

**Depth interpretation:** `crease_minus_knee_top_norm = crease_y - knee_top_y` (positive means estimated hip crease is *lower* in the image than estimated knee top). `crease_minus_knee_top_px` reports the same difference in **image pixels**, not centimetres. When the signed difference falls within ±1% of frame height, status is `near_level_uncertain`. No automatic classification in this project establishes official powerlifting depth.

The 2D transformation is anchored to the hip-to-knee segment. It follows translation, rotation and apparent length changes of that segment, but the actual anatomical surfaces can move relative to it as clothing, skin and knee articulation change. Perspective, camera angle and occlusion can also invalidate the estimate. Manually review event screenshots, especially at the bottom of each repetition.

### 4. Optional configuration

**Limit the analysis to a selected interval** (timestamps in seconds):

```bash
python app.py --input data/squat.mp4 --output outputs/isolated --start-time 35 --end-time 50
```

**Require larger movement to count a repetition** (useful if warm-up movements cause false positives):

```bash
python app.py --input data/squat.mp4 --output outputs/strict --min-knee-excursion 50 --min-hip-drop 0.10
```

**Select the athlete's visible body side:**

```bash
python app.py --input data/squat.mp4 --side right
```

**Skip annotated video export** while retaining tabular output and event screenshots:

```bash
python app.py --input data/squat.mp4 --no-video
```

Check all command-line options with:

```bash
python app.py --help
```

## Output files

```text
outputs/demo/
├── annotated.mp4          # Video overlay with body landmarks and angle guides
├── event_screenshots/     # Phase PNGs with knee-angle AND calibrated HC/KT depth guides
├── events.csv             # Event timestamps, screenshots and HC/KT depth values
├── events.json            # Event records in JSON format
├── attempts.csv           # Attempts, outcome, peak calibrated depth (if present)
├── frames.csv             # Per-frame knee/hip angles plus independent HC/KT tracking
└── summary.json           # Counts, model performance and depth calibration status
```

### Movement events

| Event | Interpretation |
|---|---|
| `descent_start` | Approximate start of the downward movement |
| `descent_end` | Approximate lowest position reached |
| `ascent_start` | First sustained upward movement detected |
| `lockout` | Estimated stable return to upright posture |
| `suspected_failure` | An apparent ascent reversal or prolonged stall without recovery, when supported by the detected movement |

Not all events are present in every attempt. For instance, an attempt that never visibly begins ascending will not have an `ascent_start` event. A video ending before the lifter stands is treated as **incomplete**, not automatically failed.

`attempts.csv` distinguishes completed attempts from incomplete or suspected failed ones. A **bottom pause** is estimated from the time between `descent_end` and `ascent_start`; short transitions below the configured minimum are reported as zero.

## Understanding the measurements

- **Knee angle:** Projected angle between hip–knee and knee–ankle line segments.
- **Hip angle:** Projected angle between shoulder–hip and hip–knee line segments.
- **Legacy hip-joint depth proxy:** Whether the estimated **hip joint landmark** drops below the estimated knee joint landmark in the image. This is not equivalent to the hip crease passing below the top of the knee.
- **Calibrated hip-crease-to-knee-top depth:** Separate tracked estimates of the two user-selected anatomical reference positions. The signed image-plane vertical separation is available per frame and at every detected critical phase. These are propagated estimates, **not directly detected anatomical surfaces on each frame**, and must be manually verified.
- **Lockout:** An estimated sustained upright posture, not a competition-rule determination.
- **Suspected failure:** A movement-pattern heuristic. It cannot determine intent, effort or the reason a lift stopped.

All measurements are affected by camera perspective, frame rate, clothing, landmark confidence and occlusion. Angles are **2D projections**, not laboratory-grade 3D biomechanics.

## Troubleshooting

| Problem | What to check |
|---|---|
| Too many reps detected | Inspect `events.csv` and screenshots; isolate the squat with `--start-time` / `--end-time`, or increase `--min-knee-excursion` and `--min-hip-drop`. |
| No reps detected | Check whether the full movement and upright posture are visible. Inspect `summary.json`; reduce thresholds cautiously. |
| Incorrect joint tracking | Try `--side left` or `--side right`; use a clearer side-view recording. |
| Crease/knee top not measured | Run `calibrate_depth.py`, then include `--depth-calibration depth_calibration.json`. |
| Crease or knee-top marker drifts | Check the preview and event screenshots; recalibrate on a clearer frame, reduce occlusion and use a more side-on camera view. |
| No annotated MP4 | Check whether OpenCV supports MP4 encoding on your system; `--no-video` still exports screenshots and data. |
| MediaPipe feedback-manager warnings | Some warnings are nonfatal if video analysis continues and output files are produced. |
| Unreliable phase timestamps | Review screenshots and compare manually with the source video; event boundaries depend on temporal thresholds. |

## Testing and validation

Run the included synthetic regression and export tests:

```bash
python -m unittest discover -s tests -v
```

Tests exercise joint geometry, calibrated surface point propagation, missing-depth handling, annotated calibrated screenshots and CSV exports, repetition segmentation, spurious bends, missing landmarks, incomplete attempts, suspected failures and event rendering. Passing them does **not** establish accuracy on real footage. The project's depth, timing, counting and failure heuristics require comparison against manually annotated squat videos before any performance claims can be made.

## Repository structure

```text
squatvision/
├── app.py                    # Command-line interface, inference and exports
├── download_model.py         # Download pretrained pose model
├── calibrate_depth.py        # Reference frame manual hip crease / knee top selection
├── squatvision/
│   ├── analysis.py           # State machine and movement events
│   ├── geometry.py           # Landmark selection and joint angles
│   ├── depth.py              # Manually calibrated anatomical depth estimate
│   └── render.py             # Visual guides and annotations
├── tests/                    # Automated regression and output tests
├── requirements.txt
├── CHANGELOG.md
├── LICENSE
└── README.md
```

## Privacy, scope and licence

Video processing runs locally after the pretrained model is downloaded. Videos, downloaded model weights and generated outputs are excluded from Git by the supplied `.gitignore`. Always check `git status` before publishing, and avoid uploading recordings of non-consenting gym users.

SquatVision is intended for **learning, experimentation and computer vision demonstrations**. Anatomical depth requires manual reference-point selection and remains an unvalidated, view-dependent estimate. It is not a validated tool for medical advice, injury prediction or official powerlifting judging.

The repository's source code is MIT-licensed. The downloaded pretrained model and dependencies are third-party components subject to their respective terms.

---

## v4: Visually tracked anatomical reference points and IPF-depth estimate

**Important migration:** The old v3 calibration stored only thigh-relative coordinates. It cannot recover the exact clicked point or visually track surface anatomy. **Recalibrate with v4**; using a v1 calibration with the new visual tracker raises an explicit error rather than silently producing unverified point trajectories.

Official [IPF Technical Rulebook (effective 1 March 2026)](https://www.powerlifting.sport/fileadmin/ipf/data/rules/technical-rules/english/2026_IPF_Technical_Rulebook__effective_01_March_2026__v3.pdf): for the squat, the top surface of the legs at the hip joint must descend below the top of the knees. SquatVision reports only an **estimated 2D vertical relationship**, not an official adjudication. A calibrated visible crease may not exactly represent the legally relevant hip-joint top surface.

### Why the old visual tracking was misleading

In v3 both manually clicked surface points were stored in a coordinate system attached to **the same MediaPipe thigh segment**. Every subsequent point was reconstructed from estimated hip and knee joint centres. It was a pose-based geometric projection, not independent anatomical tracking. The transform could theoretically move horizontally, but it did not have independent visual evidence for the actual crease or knee surface and could drift or appear nearly vertical.

v4 starts from actual clicked pixels on a chosen reference frame and uses **two independent local feature groups** with OpenCV pyramidal Lucas–Kanade flow, forward/backward flow checks, RANSAC partial-affine estimation, pose-consistency checks and an uncertainty gate. A lost marker is withheld; it is **never secretly replaced** with a pseudo-anatomical point generated from joint geometry. Additional manually selected reference frames can restore a lost track. It remains vulnerable to clothing texture movement, occlusion, incorrect point selections and oblique camera views. Tracking confidence is a quality heuristic, not a statistically calibrated probability of anatomical correctness.

### Recommended commands (Windows PowerShell)

```powershell
cd C:\path\to\squatvision
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python download_model.py

# Re-select the exact anatomical references using the upgraded calibration GUI.
python calibrate_depth.py --input data/squat.mp4 --time 71.490 --side right --output depth_calibration_v4.json

# If tracking becomes unavailable during a phase, annotate another *different* frame.
# This preserves the original anchor and adds a manually verified reinitialisation.
python calibrate_depth.py --input data/squat.mp4 --time 73.600 --side right --append depth_calibration_v4.json --output depth_calibration_v4.json

# Run only the relevant clip; the calibration reference must fall within this window.
python app.py --input data/squat.mp4 --side right --start-time 64 --end-time 80 --min-visibility 0.30 --depth-calibration depth_calibration_v4.json --output outputs/v4_right

python -m unittest discover -s tests -v
```

Interactive calibration controls: **1** selects hip crease, **2** selects knee top; click either the full frame or **4× zoom window** to set/reposition each point; `+` and `-` change zoom, **P** previews the nearby movement, **Enter/Space/S** validates and saves, **R** resets, and **Esc** cancels. Orange crosses indicate MediaPipe joint centres, **not** the anatomical surfaces. Colored circles indicate independently clicked references. The reference GUI displays distances from those joints and saves a local nearby-frames preview.

You may also supply `--hip-crease 128,274 --knee-top 202,329` for a headless calibration **only if those coordinates were actually verified on the same source frame**; they are not default ground truth.

### Depth interpretation

For source image coordinates, where Y increases downwards:

```text
depth_difference_px = hip_crease_y_px - knee_top_y_px
normalised_depth = depth_difference_px / estimated_thigh_length_px
```

- Positive means **estimated** hip crease lower in the image.
- Negative means **estimated** hip crease higher in the image.
- Near-zero values are `near_threshold_uncertain` based on a minimum 6 px tolerance or 2.5% of estimated thigh length.
- Poor optical flow, missing pose, occlusion and inadequate texture produce `uncertain`/`unavailable`, never a purported official legal-depth decision.

The `frames.csv` export retains v3 fields and adds independent **pixel X/Y**, `depth_difference_px`, `depth_difference_normalized`, per-point tracking confidence, depth status and any failure reason. The `attempts.csv` and `summary.json` include a maximum measured depth, timestamp, frame and both reference positions for each attempt when tracking is sufficiently confident; missing measurements remain missing. The `events.csv` and PNG event screenshots preserve descent, bottom, ascent, lockout and suspected-failure information.

Five dependency-free SVG plots appear in `outputs/v4_right/diagnostics/`: hip-crease X/Y over time, knee-top X/Y over time, relative depth over time, and tracking confidence over time. Open in any browser; absent data breaks lines instead of connecting missing measurements.

### Manual ground-truth evaluation

Annotate at least 5–10 frames from your **original unannotated recording** across standing, descent, bottom and ascent. Each label must contain **source-video pixels** in the following CSV structure:

```csv
frame,hip_crease_x,hip_crease_y,knee_top_x,knee_top_y
2100,128,274,202,329
```

The above row is a **format example only**; replace it with genuinely inspected annotations for each frame. Compare against the generated predictions:

```powershell
python validate_depth.py --frames outputs/v4_right/frames.csv --ground-truth manually_checked_points.csv --output outputs/v4_right/depth_validation.json
```

This reports per-point Euclidean errors, depth difference error and measurement coverage. Include unavailable frames in coverage when reporting the result. Do not claim accuracy based solely on passing synthetic tests or on MediaPipe's pose visibility score.

### Video assumptions and limitations

- A single athlete with the selected body side clearly visible; **multi-person subject re-identification is not yet implemented**. Verify that an apparent switch to another person has not occurred. When in doubt, stop interpreting the depth result.
- Source recording is constant frame rate, same resolution as calibration and fixed camera. A near-perpendicular side view reduces perspective errors; a single 2D camera still cannot establish depth with referee-level certainty.
- Point trackers follow **local image texture**, which may move relative to the true anatomical surface when shorts shift or the knee is covered. Feature groups may fail, or appear confident while tracking the wrong textured feature.
- Each manually verified extra reference can reinitialise tracking over a portion of the clip. References are not automatically fabricated after tracking loss.
- Computation makes an additional video pass and may be slower for long clips, particularly when seeking backwards from a calibration anchor. Restrict analysis to the squat with `--start-time` and `--end-time`.
- IPF validity also depends on other rules (commands, movement, stability), so **“estimated below” is not equivalent to “good lift.”**

### Validation status

Regression tests include independent horizontal/vertical trajectories with changing signed depth, insufficient visual texture, incorrect body side, persistence of phase/event exports, calibration JSON compatibility, and missing-data handling. These are **synthetic software tests**. The original workout footage was not supplied in the current execution environment; real anatomical error, actual squat depth and true rep count must be checked on the user's source recording.

**Real recording diagnostic note:** On the supplied original video, isolated texture-only flow was found to stick to a static region near the knee and lose the hip region. v4 includes a pose-motion gate to reject such outcomes rather than showing incorrect anatomical measurements. The final pose-guided tracker still requires evaluation with MediaPipe on the original file; see [`REAL_VIDEO_VALIDATION.md`](REAL_VIDEO_VALIDATION.md) for a precise validation status.

## v5.2 depth-reporting correction (important)

The earlier annotated clip reported **EST. ABOVE** from optical-flow positions
before losing the hip and knee surface tracking at the squat bottom. That was
not a valid full-repetition conclusion. v5.2 deliberately distinguishes:

- **FLOW / UNVERIFIED:** Optical-flow coordinates and a provisional pixel margin
  are available. The overlay no longer calls these IPF-compliant/non-compliant.
- **UNAVAILABLE:** Optical flow lost one or both surface points. No margin is
  shown on the affected frame. Missing data must not be counted as an above-depth
  result.
- **MANUAL / VERIFIED:** Both visible surface points were selected or verified
  against the *original, unannotated video* at that exact frame. A signed margin
  and user-chosen error tolerance are reported for that observation.
- **Attempt summary:** A sparse negative observation (even a manual one) is
  **not** a sufficient basis to declare the whole rep above required depth.
  If at least one manually annotated frame is confidently below threshold, the
  summary says **estimated below at an observed frame**, never an official ruling.

A numeric optical-flow margin can remain in `frames.csv` and `depth_vs_time.svg`
for debugging; it is **unverified**, regardless of apparent tracking confidence.
This is a conservative change, not a claim that the anatomical tracker now
operates reliably throughout the squat.

### Recommended procedure for this footage

Do not choose the initial standing or early descent frame as the sole depth
reference. Inspect several source frames around the previously detected bottom
(`72.758 s`), especially near `72.5–73.3 s`. The actual maximum relative
**surface** depth need not exactly match the knee-angle minimum.

```powershell
python manual_depth.py --input data/squat.mp4 --output data/manual_depth.json --start-time 72.5 --end-time 73.3 --uncertainty-px 8
```

Left-click the visible upper thigh at the hip, then the actual top-of-knee
surface. Repeat on several nearby frames using A/D (one frame) or J/L (ten
frames). **S** or **Enter** saves; **Esc** or the window X cancels. If clothing,
viewpoint or occlusion prevents either reference from being located, do not
invent a point; leave that frame unlabelled.

```powershell
python app.py --input data/squat.mp4 --output outputs/v52_review --side right --start-time 64 --end-time 80 --min-visibility 0.30 --manual-annotations data/manual_depth.json
```

Optionally append `--depth-calibration depth_calibration_v4.json` for
**provisional** optical-flow tracing; only manually reviewed frames are treated
as anatomical evidence for above/below status. The annotated video keeps the
angle guides but shows a shorter knee-height reference line, compact HUD and
no numerical angle labels over the thigh.

IPF depth is defined by the upper surface of the thigh at the hip compared
with the top of the knee. The visible shorts boundary is not necessarily the
underlying anatomical surface. A 2D view, manually chosen points and an
8-pixel tolerance are not validated competition judging.
