# SquatVision surface-review validation

Date: 19 October 2026 (Singapore).

## Result and limits

The patch implements better evidence collection, conservative rejection and manual
review. **It does not solve automatic anatomical localisation on the supplied
recording.** No validated automatic bottom-depth result can be reported. This is
an explicit remaining acceptance-criterion gap, not a claim of completed IPF
recognition. No thresholds were weakened to force a depth verdict.

The only paired proposal in the tested window was rejected for insufficient
nearby support. Inspection shows its hip marker follows the shorts/shirt boundary;
that is an example of potentially misleading clothing evidence, not verified
upper-thigh anatomy. Its raw pixel margin must not be used as a squat verdict.

## Inputs and reproducibility

- Baseline: supplied `squatvision_source_only.zip`, 218,115 bytes. Windows archive
  separators were normalised during extraction; bytecode caches were excluded.
- The incremental ZIP's `auto_surface.py` exactly matches the snapshot after line
  ending normalisation. It was not blindly applied a second time.
- Source: original `squat.mp4`, 21,693,072 bytes, 2,862 frames,
  29.99001729915665 FPS, upright size 352 × 640.
- Source SHA-256: `ff712650ff13d656adbdcdf85350bd800f69dd2a2fcf21b8f174ae4d604c27cc`.
- Tested runtime: Linux, Python 3.12, MediaPipe 0.10.21, OpenCV 4.11.0,
  NumPy 1.26.4; CPU inference. Windows native GUI behaviour was not tested.
- Existing official Lite Pose Landmarker model SHA-256:
  `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`.
- No independent manual ground-truth JSON/CSV or current diagnostic summary was
  available as a separate file. The pasted diagnostic specification was inspected.

Validation command (this interval is specific to this fixture, not a default):

```powershell
python app.py --input data/squat.mp4 --output outputs/surface_review --side right --start-time 64 --end-time 80 --min-visibility 0.30
```

The encoded recording is 640 × 352 with rotation metadata. This OpenCV build
initially had automatic orientation disabled, producing zero detected repetitions
in both baseline and modified code. Explicitly enabling orientation restored
portrait pixels. For an equal-pixel baseline comparison, the baseline capture was
configured with the same OpenCV orientation flag; no movement or detector
parameters were changed. The final patch uses this flag in every video consumer.

## Full-pipeline comparison on upright original pixels

| Measurement | Supplied baseline | Updated code |
| --- | --- | --- |
| Analysed frames | 480 | 480 |
| Usable pose fraction | 1.0 | 1.0 |
| Completed repetitions | 1 | 1 |
| Minimum-angle reference | Frame 2202, 73.424432 s, 71.107377° | Identical |
| Frames inspected for surfaces | 1 | 29 |
| Hip boundary candidates | None returned at reference | 5 raw candidates |
| Knee boundary candidates | None returned at reference | 4 raw candidates |
| Raw paired candidates | 0 | 1 |
| Temporally supported automatic pairs | 0 | 0 |
| Final automatic depth | Unavailable | Unavailable |

Unchanged movement events: descent start frame 2098 (69.957 s), descent end
frame 2182 (72.758 s), ascent start frame 2208 (73.624 s), lockout frame 2271
(75.725 s). Both implementations produced identical frame indices/events under
this runtime. The repetition-analysis and joint-geometry algorithms were not edited.

The reference in the user's earlier log was frame 2187 at 72.924 s / 70.04°.
This run does not reproduce those exact model outputs. The environment/model
inference differs; the code derives frame 2202 dynamically and does not hardcode
the earlier timestamp. This uncertainty must be retained when comparing runs.

The baseline reproduced `superior_thigh_edge_occluded_or_missing`. The updated
reference frame reports `POSE_MASK_MISMATCH`; per-frame evidence distinguishes
foreground support, local boundary coherence and temporal support failures.
The review window contains frames 2188–2216. Five hip proposals and four knee
proposals are retained for inspection even though no pair enters measurements.

The initial final-algorithm run measured mean 79.1 ms per candidate frame (median
74.1 ms), including seeking, decoding and second-pass inference. The surface pass
including diagnostic rendering took 2.507 seconds for 29 frames. These are one-run
CPU observations, not benchmarks or expected Windows timings. Regenerated output
metadata may report different timings.

## Numerical accuracy not established

Hip localisation error, knee localisation error, depth-margin error and false
positive rate cannot be calculated without independent anatomical reference
labels. No zero-error or accuracy percentage is claimed. Visual inspection of
frame 2201 identifies a plausible clothing-boundary false candidate; temporal
rejection prevents its raw −19.18 px difference entering per-frame depth results.
It does not prove all possible false candidates will be rejected.

The delivered validation ZIP contains the actual summary, CSVs, annotated video
and all 29 diagnostic panels. These are outputs of this recording, not substitutes
for source pixels or ground truth.

## Regression tests and review workflow

The supplied baseline had 67 discovered tests with one import error because
`validate_depth.py` was absent. The utility was restored. The final suite has
94 passing tests, including the existing movement/batch tests and new checks for:

- Dynamic per-rep windows across 24/30/60 FPS and a short multi-rep sequence.
- Independent partial proposals, torso overlap, missing boundaries, competing
  RGB bands, flat imagery, disconnected foreground and invalid geometry.
- Positive/negative margins, unsupported pairs and temporal displacement rejection.
- Explicit manual acceptance, retained repetition notes, partial-pair rejection,
  reference-frame startup, Save without implicit acceptance, and window-X cancel.
- Rotation setup, unsupported rotated backends and diagnostic source immutability.

The old single-reference test was updated to assert multi-frame behaviour.
The manual-output test was updated for per-annotation auxiliary directories.
Synthetic masks and mocked GUI tests are not anatomical accuracy validation.
No interactive Windows/high-DPI click session was performed.

## Delivery verification

Both the complete patch and code-only patch are generated from the supplied
snapshot, not GitHub. Clean-checkout `git -c core.autocrlf=true apply --check`, patch application,
source-byte comparison after normalising line endings, and the patched test suite
are checked before delivery. New source/test files are included. No GitHub push,
merge or remote modification is performed.
