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
