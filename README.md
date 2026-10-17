<<<<<<< HEAD
# SquatVision

**Video-based squat analysis using Python, MediaPipe and OpenCV.**

SquatVision is a computer vision prototype that analyses **recorded, side-view squat videos** to estimate joint angles, identify movement phases, count completed repetitions and export timestamped screenshots. It is designed as a practical demonstration of human pose estimation, temporal event detection and reproducible video analysis.

> **Important:** SquatVision is an experimental analysis tool, **not** a certified powerlifting judge, clinical assessment tool or substitute for a qualified coach. Its depth, lockout and failure labels are approximate, camera-dependent measurements.

## Features

- **Human pose estimation:** Uses the pretrained MediaPipe Pose Landmarker to estimate shoulder, hip, knee and ankle positions.
- **Joint-angle analysis:** Measures projected 2D knee and hip angles, with dashed visual guides showing the angle geometry.
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

### 3. Optional configuration

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
├── event_screenshots/     # PNG image of each detected movement event
├── events.csv             # Event names, timestamps, frames, screenshot paths
├── events.json            # Event records in JSON format
├── attempts.csv           # One record per attempt, including estimated outcome
├── frames.csv             # Per-frame measurements and smoothed values
└── summary.json           # Counts, inference performance and configuration
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
- **Depth proxy:** Whether the estimated **hip joint landmark** drops below the estimated knee joint landmark in the image. **This is not equivalent to the hip crease passing below the top of the knee.**
- **Lockout:** An estimated sustained upright posture, not a competition-rule determination.
- **Suspected failure:** A movement-pattern heuristic. It cannot determine intent, effort or the reason a lift stopped.

All measurements are affected by camera perspective, frame rate, clothing, landmark confidence and occlusion. Angles are **2D projections**, not laboratory-grade 3D biomechanics.

## Troubleshooting

| Problem | What to check |
|---|---|
| Too many reps detected | Inspect `events.csv` and screenshots; isolate the squat with `--start-time` / `--end-time`, or increase `--min-knee-excursion` and `--min-hip-drop`. |
| No reps detected | Check whether the full movement and upright posture are visible. Inspect `summary.json`; reduce thresholds cautiously. |
| Incorrect joint tracking | Try `--side left` or `--side right`; use a clearer side-view recording. |
| No annotated MP4 | Check whether OpenCV supports MP4 encoding on your system; `--no-video` still exports screenshots and data. |
| MediaPipe feedback-manager warnings | Some warnings are nonfatal if video analysis continues and output files are produced. |
| Unreliable phase timestamps | Review screenshots and compare manually with the source video; event boundaries depend on temporal thresholds. |

## Testing and validation

Run the included synthetic regression and export tests:

```bash
python -m unittest discover -s tests -v
```

Tests exercise joint geometry, repetition segmentation, spurious bends, missing landmarks, incomplete attempts, suspected failures and event rendering. Passing them does **not** establish accuracy on real footage. The project's depth, timing, counting and failure heuristics require comparison against manually annotated squat videos before any performance claims can be made.

## Repository structure

```text
squatvision/
├── app.py                    # Command-line interface, inference and exports
├── download_model.py         # Download pretrained pose model
├── squatvision/
│   ├── analysis.py           # State machine and movement events
│   ├── geometry.py           # Landmark selection and joint angles
│   └── render.py             # Visual guides and annotations
├── tests/                    # Automated regression and output tests
├── requirements.txt
├── CHANGELOG.md
├── LICENSE
└── README.md
```

## Privacy, scope and licence

Video processing runs locally after the pretrained model is downloaded. Videos, downloaded model weights and generated outputs are excluded from Git by the supplied `.gitignore`. Always check `git status` before publishing, and avoid uploading recordings of non-consenting gym users.

SquatVision is intended for **learning, experimentation and computer vision demonstrations**. It is not a validated tool for medical advice, injury prediction or official powerlifting judging.

The repository's source code is MIT-licensed. The downloaded pretrained model and dependencies are third-party components subject to their respective terms.
=======
# squatvision
SquatVision is a Python-based computer vision project using MediaPipe and OpenCV to analyse squat biomechanics, estimate joint angles and depth, track repetition phases, detect pauses, and generate timestamped movement analysis reports.
>>>>>>> e8101716a7a221587c6966e2310bfbf49bd5c9b9
