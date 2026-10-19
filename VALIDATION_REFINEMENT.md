# SquatVision — uploaded source and original-video review

Date: 19 October 2026.

## Inputs inspected

- Latest uploaded source ZIP: `SquatVision_source_only(1).zip` (39 relevant files).
- Original unannotated video: `squat(1).mp4`, 21,693,072 bytes, 2,862 frames,
  29.990 FPS, decoded upright at 352×640; SHA-256
  `ff712650ff13d656adbdcdf85350bd800f69dd2a2fcf21b8f174ae4d604c27cc`.
- Existing uploaded diagnostic screenshots and prior `pose_mask_v4` textual
  summaries show one completed repetition near 72.9 seconds, with frame 2188
  providing an **unverified** hip/knee pair and no temporal corroboration.

## What was implemented

- Kept core repetition detection, knee-angle reference selection, MediaPipe
  segmentation invocation, manual annotation data format, and strict IPF
  measurement provenance unchanged.
- For curved **knee silhouette** outlines, permitted a tightly bounded quadratic
  curve when the straight-line fit fails. It still returns the physically
  observed centre ray, not a fitted or extrapolated guess. For hip boundaries,
  internal RGB boundaries, or irregular evidence the original rejection logic
  is preserved. This is a geometry refinement, not body-part segmentation.
- Retained rejected raw candidate depths only under
  `raw_candidate_margin_px`; cleared `margin_px` for rejected automatic pairs.
- Added magnified hip/knee source and segmentation crops at each evaluated
  reference-window frame to support inspection of garment/torso ambiguity.
- Added optional `--open-best-candidate` review navigation; selecting a
  review frame does not certify landmarks or imply a depth outcome.

## Local verification performed

- Python source compiles.
- Baseline from the supplied ZIP: **100 tests passed**.
- After incremental changes: **109 tests passed**, including 9 new tests of
  rejection semantics, review frame selection, knee silhouette curvature,
  crop boundaries, missing masks, RGB-edge uncertainty and original-array
  immutability.
- Original MP4 was successfully decoded, and full-length video metadata and
  frame samples were inspected. No MediaPipe runtime or Pose Landmarker model
  weights are installed in this sandbox; therefore, **no full inference run was
  performed on the original video after this patch**. No claim of improved
  automatic detection on that video, anatomical localisation accuracy,
  IPF compliance or clinical reliability is justified.

## Windows validation

From the SquatVision project directory after applying the code:

```powershell
python -m unittest discover -s tests -v
python app.py --input data/squat.mp4 --output outputs/review_refinement --side right --start-time 64 --end-time 80 --min-visibility 0.30
$report = Get-Content outputs/review_refinement/summary.json -Raw | ConvertFrom-Json
$report.auto_bottom_reference_frames | Select-Object rep_number,frame,status,reason,valid_proposal_frames,suggested_review_frame | Format-List
python manual_depth.py --input data/squat.mp4 --analysis-summary outputs/review_refinement/summary.json --rep-number 1 --open-best-candidate --output data/squat_manual.json
```

The original source name in this archive upload is `squat(1).mp4` only because
of the upload system. The commands above expect the existing local
`data/squat.mp4` and its matching SHA-256, as recorded in the diagnostics.

## Limitations and next objective

This improvement does **not** train or integrate a new human-part model. The
whole-person mask cannot reliably separate an upper-thigh anatomical surface
from shirt/shorts or torso overlap. At least 5–10 manually verified frames
around the bottom, preferably a less occluded high-resolution side view as a
second sample, are needed to quantify localisation error and decide whether
human parsing meaningfully improves the detection. Preserve `UNAVAILABLE`
when the upper thigh is not visible rather than forcing a judgement.
