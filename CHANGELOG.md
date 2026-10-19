# SquatVision — Changelog

> **Reconstructed and reconciled 20 October 2026.** This document replaces fragmented, duplicated version notes in earlier snapshots. It describes the project's development, not a sequence of independently released packages. **The latest review-refinement ZIP is a prepared working snapshot; it has not been confirmed as installed or pushed to GitHub.** Version labels below identify historical development milestones, while subsequent named patches are described separately rather than assigned invented release numbers.
>
> **Current scope:** Video-based 2D hip/knee joint-angle analysis, squat repetition and phase detection, and **experimental** comparison of upper-thigh-at-hip and knee-top *surface* heights. SquatVision is **not** an IPF-certified judging system. Missing/occluded landmarks do not establish inadequate depth.

## At a glance — what works, and what does not

| Capability | Current state | Interpretation |
| --- | --- | --- |
| MediaPipe hip and knee **joint-centre** angles | **Working on reference video** | Two-dimensional, camera-dependent angles; not the IPF surface-depth criterion. |
| Completed-rep counting and phase/pause timestamps | **Working on reference video** | One completed rep in the diagnostic 64–80 s interval; phase timestamps remain approximate. |
| Per-repetition minimum knee-angle frame | **Working on reference video** | Frame **2187**, **72.924 s**, approximately **70.04°** for the original `squat.mp4`. This is a *review reference*, not proven maximum anatomical depth. |
| Video outputs, events, summaries, batch support | **Implemented** | Subject to input, installation and runtime constraints. |
| Manually selected hip-surface / knee-top points | **Implemented; user review required** | Can yield signed image-plane measurements on labelled frames, subject to point accuracy and camera perspective. |
| LK optical-flow propagation from selected surfaces | **Experimental; unreliable on this footage** | Can attach to background/clothing or lose points; a tracked pixel is not verified anatomy. |
| Automatic surface proposals from segmentation / RGB boundaries | **Implemented but unsuccessful on reference video** | Latest confirmed Windows run evaluated **29** nearby frames and accepted **0** temporally supported hip/knee pairs; the one raw pair was rejected. |
| Automatic IPF below-parallel decision | **Not achieved / not validated** | `UNAVAILABLE` is the correct outcome where anatomical evidence is insufficient; do **not** infer a failed squat. |

**Two different measurements must never be conflated:**

- **Joint-angle analysis:** MediaPipe's shoulder/hip/knee/ankle **joint centres** estimate hip and knee angles, movement and repetitions.
- **IPF-style surface comparison:** The **top of the thigh at the hip joint** must be compared with the **top of the knee**. With image coordinates increasing downwards, `margin_px = hip_surface_y_px - knee_top_y_px`: positive means the *estimated image-plane* thigh reference is lower. It is neither a centimetre measurement nor an official ruling. The automatically estimated points may be clothing contours rather than the required anatomy.

---

## Working-copy updates (not new formal releases)

### 2026-10-19 — Review-evidence refinement (latest prepared snapshot; integration status unconfirmed)

**Goal:** Make uncertain automatic results easier to inspect without claiming improved anatomical accuracy.

- Added original-pixel, enlarged **hip and knee close-up panels** alongside whole-frame diagnostic images and segmentation-mask visualisations.
- Added `suggested_review_frame` / `suggested_review_reason` and an optional `manual_depth.py --open-best-candidate` workflow. Selecting a frame is **not** accepting its proposed landmarks.
- Separated `raw_candidate_margin_px` (a rejected or unverified proposal) from the accepted `margin_px` field in reports/CSV. A raw signed number is not automatically a valid depth measurement.
- Added a narrowly constrained **curved-silhouette fallback for the knee only** when a straight-line contour fit fails. Kept stricter checks for the hip, internal RGB edges, and occluded anatomy.
- Added **9** deterministic tests: **100 baseline → 109 passing** in the dependency-light development environment.
- Reviewed the uploaded original video and prior diagnostics; **did not** perform new end-to-end MediaPipe inference in that environment, because MediaPipe/model weights were unavailable. Do **not** infer increased real-video depth accuracy from the test count.
- Source of truth for these changes: `SquatVision_review_refinement_source.zip` and `VALIDATION_REFINEMENT.md` (19 October). **Whether this snapshot has since been installed in the Windows checkout is unknown.**

### 2026-10-19 — Pose–segmentation association refinement (confirmed on reference recording)

**Problem:** Re-running MediaPipe on a cropped ROI changed estimated joint coordinates significantly; the earlier hard `SEGMENTATION_POSE_MISMATCH` gate rejected many bottom frames despite a foreground mask covering the original tracked athlete.

- Kept the **original pose pass** authoritative for angles and motion phases.
- Added original-pose-to-mask checks (hip, knee and intervening thigh samples on the same foreground component); recorded cropped-pass disagreement as diagnostic evidence rather than automatically rejecting moderate differences.
- Retained safeguards for clear identity/foreground mismatches; this is **not** general multi-person tracking or subject re-identification.
- **Observed Windows result:** The original pose was supported by the mask in **29/29** reviewed bottom-window frames. The minimum-angle frame changed from a pose mismatch to `HIP: NO_SUPPORTED_INTERNAL_BOUNDARY`.
- **Still unsuccessful:** Only **one** raw frame contained both proposed points (frame **2188**, **72.958 s**, about **−9.56 px**); `INSUFFICIENT_VALID_FRAMES` correctly rejected it, leaving **0** accepted automatic pairs and overall depth `UNAVAILABLE`. The number is a *raw proxy*, not an IPF judgement.

### 2026-10-19 — MediaPipe mask-crash and interface fixes (confirmed on reference recording)

**Problem:** The new secondary segmentation pass initially caused a native MediaPipe assertion, `Check failed: 1 == ChannelSize() (1 vs. 4)`. Subsequent code edits introduced an undefined `detected` variable and tests exposed outdated MediaPipe mocks.

- Restored the required `detected = landmarker.detect(img)` inference call before using segmentation output.
- Padded RGB crop widths to a multiple of four where necessary; checked `mask_image.is_contiguous()` **before** calling `.numpy_view()` to avoid a known unsafe mask conversion path.
- Accepted MediaPipe singleton-channel masks of shape `(H, W, 1)` as well as `(H, W)`, removed the singleton channel, and stripped only the temporary right-side padding before inserting the mask into the original frame.
- Updated synthetic MediaPipe tests to model mask contiguity, pose-landmark coordinates and the current `open_video()` path; preserved rejection of non-contiguous masks.
- **Observed Windows result:** The program stopped crashing, analysed **480** selected frames, detected **1** completed rep, and proceeded into anatomical boundary evaluation. TensorFlow Lite feedback-manager messages remained **nonfatal warnings**.
- **Still unsuccessful:** Correctly shaped masks did not themselves locate the IPF anatomical surfaces.

### 2026-10-19 — Multi-frame surface review (applied and tested on reference recording)

**Problem:** The first automatic-bottom approach searched a whole-person silhouette for an exposed superior-thigh edge; when the torso and thigh merged, it returned `superior_thigh_edge_occluded_or_missing`.

- Introduced `squatvision/surface_evidence.py` and video-handling improvements; evaluated **hip and knee candidates independently** using pose-guided regions, whole-person masks and internal image-colour boundaries.
- Inspected multiple frames around each reference (nominal **±0.5 s**, constrained by rep bounds), rather than requiring a single reference frame to contain both visible landmarks.
- Introduced temporal-pair consistency checks, explicit rejection reasons (`NO_SUPPORTED_INTERNAL_BOUNDARY`, `INCOHERENT_LOCAL_BOUNDARY`, `POSE_MASK_MISMATCH`, etc.), and per-frame source/mask diagnostic JPGs and JSON/CSV evidence.
- Improved manual review to open near the detected repetition reference, show partial unverified proposals, support explicit acceptance/correction, and keep source-video fingerprint checks. Added rotation-aware video capture and updated dependency guidance.
- Preserved batch processing and the existing angle/rep detector; `--manual-only` suppresses automatic proposals and propagation.
- **Observed Windows result:** One correct rep and bottom reference, **29** frames examined, **0** automatically supported anatomical pairs. Some individual hip/knee boundaries were found, but not a reliable matched pair.
- No body-part semantic segmentation model was introduced: the mask remained a **whole-person** mask, and internal RGB edges could follow a shirt/shorts boundary.

### Around 2026-10-18–19 — Automatic per-repetition bottom-reference patch (applied)

**Problem:** Earlier manual-depth examples defaulted to **72–74 s**, a window applicable only to the original reference recording, and the previous phase event (`descent_end`) was being treated as if it necessarily marked the minimum knee angle.

- Dynamically selected the **minimum valid measured knee angle** between each repetition's descent start and ascent start; recorded the actual frame index and video timestamp per attempt.
- Added the `auto_bottom_reference_frames` summary and automatically selected review screenshots; avoided hardcoded timestamps for new videos.
- Kept **minimum knee angle distinct from maximum observed surface-depth margin**. The frame at maximum hip drop need not equal either.
- First diagnostic of original recording: **1** completed rep; movement bottom event approximately **72.758 s**; minimum-angle reference **72.924 s**, **70.04°**, frame **2187**.
- The automatic silhouette proposal still failed with `superior_thigh_edge_occluded_or_missing`; **0 of 15** frames around the movement-bottom event contained a usable surface-depth measurement.
- This milestone solved **frame selection**, **not automatic IPF landmark localisation**.

---

## Historical version milestones

### v5.3 — Manual annotations seed visual tracking (October 2026; historical patch)

- Fixed the case where `--manual-annotations` displayed selected surface points only on the exact clicked frames when no separate calibration was supplied.
- Used suitable manually labelled frames as **in-memory, independent hip/knee optical-flow anchors**; maintained exact manual annotations and did not interpolate missing visual evidence.
- Added `--manual-only`, `manual_seeded_visual_tracking`, `anatomical_tracking_valid_frames`, tracking warnings and detailed unavailable-reason summaries.
- Added synthetic moving-image and invalid-anchor regression tests. Real-video anatomical tracking remained experimental and was not verified end-to-end in the patch's build environment.

### v5.2 — Conservative depth-reporting correction (historical note dated 2026-10-18)

- Corrected misleading **`EST. ABOVE`/`EST. BELOW`** overlays derived from provisional optical-flow points. Such points could be lost at the actual squat bottom and could not justify a whole-rep depth decision.
- Kept automated visual-flow estimates **unverified**, explicitly reported missing frames, and separated manually reviewed observations from automated tracking.
- Ensured one early negative margin does **not** imply that the entire squat failed the depth requirement; a single observed reliable below-threshold frame also does not constitute an official judging decision.
- Improved readability of the **352 px portrait** overlays and diagnostic depth graph; added tracking-loss/sparse-manual tests.

### v5.1 — Manual annotation GUI fixes (October 2026)

- Fixed OpenCV manual-depth editor responsiveness: each click redraws immediately; native window-close and Esc cancel without saving; S/Enter/Space explicitly save.
- Kept cursor-to-image mapping stable in an autosized window; tested mouse interactions and partial/cancelled edits.
- This fixed **annotation usability**, not the accuracy of automatic surface-point detection.

### v5.0 — Separate experimental IPF surface-depth measurement (October 2026)

**Major scope change:** Hip/knee **joint angles alone are not the IPF below-parallel rule**. v5 introduced a separate measurement stream for hip-area upper-thigh surface versus top of knee.

- Added `squatvision/ipf_depth.py` for signed and projected-thigh-normalised image-plane margins, provenance/status and engineering uncertainty; neither normalised depth nor pixels is a real-world distance.
- Added `manual_depth.py` for frame-by-frame source-video surface selection, editing, zoom, JSON/CSV and screenshots; it can operate when optical flow and MediaPipe-driven automatic depth fail.
- Added `squatvision/ipf_plot.py` and depth-vs-time visualisations that retain gaps when observations are unavailable.
- Integrated manual landmarks, optional visual tracking and surface-related overlays/exports into `app.py` while preserving joint-angle output, phases and repetition counting.
- Added regression coverage (historical patch documented **15** new tests on top of **26** earlier v4 tests). Synthetic tests did not establish IPF anatomical accuracy.

### v4 — Independent optical flow, multi-anchor recovery and diagnostics (October 2026)

- Replaced earlier **pose-relative propagation** of manually calibrated surface points with independent local Lucas–Kanade optical flow from **actual selected source-image pixels**.
- Added texture/consistency checks, pose-motion gating, multi-anchor manual calibration and recovery, and per-point uncertainty/tracking diagnostics.
- Added zoomable/manual reference tools, projected-thigh normalisation, depth plots and `validate_depth.py` for comparison against annotated references.
- Real-footage follow-up found visual tracking could **attach to background near the knee** or **lose the low-texture hip reference**. Safeguards were added to reject such results, but continuous accurate surface tracking was not proven.
- A development milestone reported **26 passing automated tests**. These were mainly software/synthetic tests.

### v3 / 3.0.0 — Manual surface calibration and pose-relative tracking (October 2026)

- First attempted to move beyond pure hip/knee joint angles by **manually specifying hip-crease/upper-thigh and knee-top image points** and propagating them relative to the observed thigh/pose (`squatvision/depth.py`).
- Added signed vertical depth indicators, dashed depth guides, calibration preview and related CSV/JSON fields alongside angle/phase exports.
- Important limitation: **moving the hip/knee joint-centre geometry does not reliably track separate visible anatomical surfaces**. Relative point propagation can drift when the athlete translates, clothing moves, or the thigh rotates/occludes.
- The older changelog gives `3.0.0` the date **2026-10-18**, but this conflicts with the surviving v4/v5 patch sequence. **The exact initial v3 build date is unverified**; the date has not been silently treated as an authoritative release timestamp.

### v2 / 2.0.0 — Robust joint-angle and repetition-analysis baseline (2026-10-16)

**First clearly documented, Git-committed functional milestone.**

- Used MediaPipe **shoulder–hip–knee–ankle joint centres** to compute projected 2D hip/knee angles; used OpenCV for annotated output.
- Reworked rep detection as a temporal state machine requiring stable upright posture, meaningful flexion/hip displacement, and return to sustained upright. Kept incomplete and suspected failed attempts separate from completed reps.
- Added descent/ascent/return-to-upright events, pause estimates, optional clip start/end time and selected side, dashed angle guides/arcs, annotated MP4, CSV/JSON and event screenshots.
- Added regression tests for multiple reps, jitter, shallow attempts, tracking loss and failed/incomplete attempts.
- Original footage initially exposed failure to count a squat under some default side/time/visibility settings. The reference run with `--side right --start-time 64 --end-time 80 --min-visibility 0.30` subsequently detected **one completed rep**. These timestamps are **reference-video-specific**, not application defaults.
- Git anchor: [initial v2 code commit](https://github.com/XJie97/squatvision/commit/b0832cfd3300a214e47f322b46a5967129ccbf06). The [early surface-calibration commit](https://github.com/XJie97/squatvision/commit/e16f1daaf7897cd581f0c399da58200db6b990fe) documents the next functional direction; later local v4/v5 patches may not be present in GitHub history.

### Initial angle-only prototype (before documented v2; exact version/date not recoverable)

- Began as a side-view computer-vision demonstration using pose-estimated **hip and knee reference angles** to describe squat movement/depth approximately.
- Angle-based depth or the vertical relation between **joint centres** was only a coarse form proxy. It **cannot** establish the IPF comparison between the top of the thigh at the hip and the top of the knee.
- No defensible original v1 commit/date or independently measured anatomical-depth accuracy was found in the available source history. This entry records **project intent**, not a separately verified release.

---

## Reference recording and validation ledger

These figures are **for the original `squat.mp4` only**, not universal expected outputs:

- Input approximately **352 × 640**, **29.990 FPS**; SHA-256 `ff712650ff13d656adbdcdf85350bd800f69dd2a2fcf21b8f174ae4d604c27cc`.
- Validated movement-analysis run on Windows: **480** analysed frames in the **64–80 s** interval; **1** detected completed repetition, **1** attempt, `selected_side=right`, `usable_pose_fraction=1.0`.
- Events: descent start approximately **70.057 s**, descent end/movement bottom **72.758 s**, ascent start **73.591 s**, return to standing **75.759 s**.
- Minimum measured knee angle in that rep: approximately **70.04°** at **frame 2187 / 72.924 s** (distinct from the phase event). Reported attempt-minimum summary angles can differ slightly because of smoothing/selection logic.
- First automatic reference: `superior_thigh_edge_occluded_or_missing`, **0** depth observations in a 15-frame bottom window.
- Surface-review iterations: **29** frames evaluated near the knee-angle reference; various independent **hip or knee** candidates, **only one raw paired proposal** at frame **2188**, **no temporally accepted surface-depth pairs**. Raw image-plane margin approximately **−9.56 px**, rejected as `INSUFFICIENT_VALID_FRAMES`; **never** a validated depth decision.
- Pose-mask-consistency patch: original pose-mask support on **29/29** evaluated frames; improved diagnosis from `SEGMENTATION_POSE_MISMATCH` to missing/ambiguous internal anatomical boundaries. Automatic depth **still `UNAVAILABLE`**.
- Latest review-refinement ZIP: **109 tests passed** locally, **but full MediaPipe end-to-end inference was not run after that patch**. Passing software tests are not ground-truth localisation results.

### Known limitations and unresolved engineering work

1. Whole-person segmentation distinguishes foreground/background, **not thigh vs torso vs clothing**. Internal colour-edge detection can mistake shirt/shorts boundaries, seams or shadows for anatomical landmarks.
2. A second cropped MediaPipe pass can move joint estimates relative to full-frame tracking; mask support improves association but does not prove anatomical identity or robust re-identification of people.
3. Optical flow can track stable but **wrong** textures; its confidence is not an anatomical accuracy probability. Missing surface points must remain missing.
4. Resolution, viewpoint and partial occlusion in the supplied footage make the **upper thigh at the hip joint** difficult to locate precisely; minimum knee flexion is not necessarily maximum verified depth.
5. Next validation prerequisite: **5–10 manually reviewed source frames around the bottom**, including both visible surface landmarks where identifiable; compare automatic candidates against these. Only then evaluate whether a separate human-part segmentation model improves real localisation.
6. The code **must not** turn a rejected raw margin, a joint angle, or a missing measurement into an IPF pass/fail label.

## Provenance and document-maintenance notes

- Reconstructed from the repository's initial commits; available v3–v5 patch artifacts and existing README/changelog snapshots; Windows diagnostic logs; uploaded source ZIP and original MP4; and the SquatVision development discussions available in this conversation's context.
- A full searchable export of **every historical ChatGPT conversation** was **not available**. Some original dates, precise local commits and the installation status of the latest ZIP are therefore unverified; the record deliberately avoids inventing them.
- GitHub's accessible commits represent the **earlier baseline**, not necessarily the latest local changes. Avoid equating a ChatGPT-generated patch with a committed release until `git status`, local tests, real-video diagnostics and the chosen branch are checked.
- **Documentation reconciliation only (20 October 2026):** No changes to Python algorithms, model weights or videos are made by this changelog rewrite. Update this file again only when code changes are actually applied and tested; distinguish **Prepared**, **Applied**, **Tested (synthetic)** and **Tested (original video)**.
