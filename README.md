# SquatVision — surface-review update

Offline, single-athlete squat analysis with separate joint-angle and experimental
surface-depth measurements. Run `python app.py` to process every MP4 directly
inside `data/`, with independent outputs under `outputs/<video-stem>/`.

**Automatic anatomical recognition remains unverified.** This update provides
visible-boundary proposals, explicit rejection reasons and manual review. On the
supplied recording it detects the squat, but does not establish an automatic
hip/knee surface-depth measurement at the bottom. See `VALIDATION_REPORT.md`.

## Setup (Windows PowerShell)

Use Python 3.12 in a clean environment for the tested dependency versions:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe download_model.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The commands below use `python` from that environment. `requirements.txt` retains
broader dependency ranges; results may differ across MediaPipe versions. Do not
install multiple OpenCV distributions into the same environment. Windows manual
review needs the GUI-enabled `opencv-contrib-python`, not a headless wheel.
No additional segmentation network or model weights are introduced. The existing
`download_model.py` retrieves the official MediaPipe Lite Pose Landmarker asset.

## Analyse recordings

```powershell
python app.py
python app.py --input-dir data --output outputs --side auto
python app.py --input data/squat.mp4 --output outputs/squat --side right
```

Choose the camera-facing side if automatic selection chooses the wrong limb.
For each attempt, the minimum valid measured knee angle between descent start and
ascent start becomes the reference frame. Missing phase events fall back to the
attempt bounds and set `phase_bounds_complete=false`. Reference times are derived
from each recording; no squat timestamps are hardcoded.

All available frames within the reference time ±0.5 seconds are inspected,
clipped to the attempt and analysed video bounds. Configure this using
`--bottom-window 0.75` (allowed range: greater than 0, at most 5 seconds).
The minimum knee angle is not necessarily the greatest surface-depth margin.
`maximum_observed` selects the largest supported observed margin separately.
Missing frames cannot prove that the squat stayed above depth.

`--start-time` and `--end-time` restrict analysis when needed. `--no-video` skips
MP4 export. `--no-auto-bottom` disables automatic surface proposals. `--manual-only`
disables both propagation and automatic proposals, retaining exact manual frames.
Batch calibration/annotations must be separate `<video-stem>.json` files:

```powershell
python app.py --annotations-dir data/annotations --calibrations-dir data/calibrations
```

No annotations, bottom timestamps or calibration coordinates are transferred
between recordings. Existing annotation files are checked against video name,
dimensions and FPS. Summary-driven review additionally checks a source SHA-256.
MP4 rotation metadata is explicitly enabled before dimensions or frames are read
in every capture path. An unsupported rotated input fails with an explanation.
Inputs are assumed constant-frame-rate; timestamps use frame index / FPS.

## What surface detection actually does

1. Re-detect the tracked athlete in a padded pose-guided ROI on each candidate
   frame. Compare the re-detected whole pose with the tracked pose to reject a
   likely subject mismatch. Keep the original joint centres for all angles.
2. Use the whole-person probability mask to check foreground/component support.
3. Test hip and knee regions independently along the upward thigh normal. Five
   local rays must support a coherent contour. The central ray supplies a directly
   observed coordinate; the detector does not extrapolate a distant edge to the hip.
4. If no outer silhouette is visible, search for an internal Lab-colour boundary
   with foreground on both sides. Reject competing bands, incoherent contours,
   out-of-image ROIs and insufficient foreground support.
5. Require at least two nearby paired candidates within 0.15 seconds whose
   pose-relative offsets agree within 0.12 projected thigh lengths. Preserve
   rejected candidates for review without filling gaps or creating measurements.

These are engineering heuristics, not calibrated probabilities or a human-part
classifier. The RGB fallback can detect a waistband, hem, shadow or overlapping
person. Stable optical flow or temporal agreement cannot prove anatomical identity.
The mask shown in diagnostics is a **whole-person mask**, not labelled body parts.

Human-parsing alternatives were evaluated against this task: SCHP LIP/ATR include
clothing labels, and Pascal-Person-Part separates upper legs from torso, but none
provides the exact required hip surface or knee-top landmark. The latter remains
an appropriate future experiment, requiring separate model integration and manual
ground-truth evaluation. It is not included or represented as implemented here.
Sources: [MediaPipe Pose](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)
and [SCHP label sets and implementation](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing).

## Review and correct proposals

```powershell
python manual_depth.py --input data/squat.mp4 --analysis-summary outputs/squat/summary.json --rep-number 1 --output data/annotations/squat.json
```

The editor opens at the exact automatically selected reference frame, within a
±1-second navigation window. Optionally add `--open-best-candidate` to start at
the most informative *raw* proposal in the neighbourhood (e.g., the paired but
rejected frame 2188 in the reference recording). This changes only where manual
review starts; it does not accept, verify or publish the proposed landmarks.
Yellow diamonds are unverified proposals. A partial proposal remains visible
even if its partner is missing. Rejected pairs can be
reviewed, but are never saved automatically.

| Action | Control |
| --- | --- |
| Previous/next frame | A / D |
| Skip ten frames | J / L |
| Accept the displayed pair after visual review | U |
| Replace with manually positioned points | Click hip surface, then knee top |
| Clear label and dismiss proposal on this frame | R / right-click |
| Zoom | + / - |
| Save completed annotations and exit | S / Enter / Space |
| Cancel without writing | Esc / window X |

A lone surface cannot be accepted with U; supply both points manually if they are
visible. Accepted and repositioned annotations use `MANUAL_ANNOTATION` provenance
and retain a repetition note. Pressing Save does not accept displayed proposals.
Manual annotation expresses a user's selection, not an independent anatomical
certification. Do not mark hidden anatomy merely to obtain a number.

The annotation JSON remains compatible with existing manual-seeded optical flow:

```powershell
python app.py --input data/squat.mp4 --output outputs/squat_reviewed --side right --manual-annotations data/annotations/squat.json
```

Manual positions override automatic results at those exact frames. Propagated
optical flow remains provisional. Use `--manual-only` to retain only exact labels.
Auxiliary screenshots, CSV, JSON and depth graph are saved in
`<annotation-stem>_review/` beside the annotation file, avoiding cross-video clashes.

## Outputs and interpretation

- `annotated.mp4`, `frames.csv`, `events.csv/json`: existing movement exports plus
  surface measurements/provenance; partial proposals appear as separate diamonds.
- `summary.json`: movement results and `auto_bottom_reference_frames`, each with
  reference frame, phase-bound completeness, all candidate rows, evaluated count,
  valid-pair count, reference status and maximum supported observed margin.
  `suggested_review_frame` and `suggested_review_reason` prioritise a useful
  image for manual inspection without making a depth judgement.
- `surface_proposals.json/csv`: raw candidate evidence. `raw_candidate_margin_px`
  retains the signed difference of a rejected raw pair, whereas `margin_px`
  is **null on rejected automatic pairs** and only populated on supported
  automatic pairs or previously annotated/tracked evidence. An isolated raw
  value (such as frame 2188's −9.56px) is **not** an accepted depth measurement.
- `surface_diagnostics/rep_XX_frame_XXXXXX.jpg`: enlarged source, whole-person mask,
  joint crosses, selected local contours, candidate circles and rejection reason.
- `surface_diagnostics/rep_XX_frame_XXXXXX_closeups.jpg`: two magnified joint-centred
  SOURCE crops plus their mask overlays, all from **original video pixels**.
  Crosses show first-pass joint centres; circles show tentative visible edges.
  Enlargement aids inspection but cannot reconstruct a hidden anatomical surface.
- `depth_vs_time.svg`: generated when manual, calibrated or supported automatic
  measurements exist. Missing values stay gaps.
- Batch root: `batch_summary.csv/json`, independent success/failure per recording.

| Candidate status | Meaning |
| --- | --- |
| `AUTOMATIC_CANDIDATE` | Raw pair before temporal filtering |
| `AUTOMATIC_UNVERIFIED` | Supported visible-boundary pair; anatomy not verified |
| `MANUAL_VERIFIED` | Exact manually reviewed annotation in candidate diagnostics |
| `EXISTING_UNVERIFIED` | Existing provisional tracked measurement preserved |
| `UNAVAILABLE` | No usable paired observation; may retain raw points for inspection |

Legacy top-level reference status `UNVERIFIED_SURFACE_PROPOSAL` means the window
has a usable proposal; inspect `reference_status` for the exact minimum-angle
frame. Existing `AUTOMATED_TRACKING` / `MANUAL_ANNOTATION` provenance remains in
measurement exports. Automatic measurements never receive an above/below verdict
in the application's strict reporting path.

Signed margin = hip surface Y − knee-top Y. Image Y increases downward: positive
is below the knee reference, negative above. Pixels are not centimetres.
Optional normalisation divides by projected thigh length. Uncertainty bounds are
engineering tolerances, not statistically calibrated confidence intervals.

## Focused review improvements (latest incremental patch)

The knee's visible *person-silhouette* may be smoothly curved. The detector now
allows a tightly constrained quadratic local fit for that distal silhouette
when the linear fit rejects a smooth knee cap. The fit does **not** move the
observed centre-ray intersection or relax the RGB/clothing-edge rules, and it
does not add semantic anatomical recognition. An occluded hip stays unavailable.

Raw paired candidates that lack temporal support retain their individual hip
and knee pixel proposals for visual review, but `margin_px` becomes `null`.
This prevents a rejected signed difference being mistaken for a valid IPF
measurement. Synthetic tests cover this invariant. No official IPF verdict is
produced by this change.

Review command for this exact recording (other videos use their own detected
frames, not 72-second defaults):

```powershell
python app.py --input data/squat.mp4 --output outputs/squat_review --side right --start-time 64 --end-time 80 --min-visibility 0.30
python manual_depth.py --input data/squat.mp4 --analysis-summary outputs/squat_review/summary.json --rep-number 1 --open-best-candidate --output data/squat_manual.json
```

The GUI requires a desktop window. Verify both anatomical points visually
before accepting a displayed pair with `U` or replace them by clicking; neither
method can reveal landmarks genuinely hidden beneath clothing.

## Troubleshooting and limitations

- `NO_SUPPORTED_INTERNAL_BOUNDARY`: no clear internal edge; review neighbouring
  frames. Whole-person masks cannot reveal hidden anatomy.
- `ANATOMICAL_BOUNDARY_AMBIGUOUS` / `INCOHERENT_LOCAL_BOUNDARY`: competing or uneven
  local edges, potentially clothing, foreground interference or occlusion.
- `POSE_MASK_MISMATCH` / `SEGMENTATION_POSE_MISMATCH`: foreground does not support
  the selected pose, or re-detection disagrees substantially. Inspect mask/side.
- `INSUFFICIENT_VALID_FRAMES` / `TEMPORAL_INCONSISTENCY`: paired evidence lacks
  nearby support. Coordinates are not interpolated to conceal the gap.
- `PROJECTED_THIGH_TOO_SHORT` / `THIGH_TOO_VERTICAL`: unsuitable projected geometry.
- Rotation error: convert to an upright CFR video and create new annotations.
- Front/rear views and oblique cameras can distort or hide the relevant surfaces.
  Outputs from several angles are independent experiments, not interchangeable rulings.

Independent localisation error requires manually reviewed pixel references:

```powershell
python validate_depth.py --predictions outputs/squat/frames.csv --references references.csv --output outputs/squat/validation.json
```

Reference columns: `frame,hip_crease_x,hip_crease_y,knee_top_x,knee_top_y` in
upright source-image pixels. No ground-truth references are bundled. Synthetic
regression tests do not establish real-video anatomical accuracy.
