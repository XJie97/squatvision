"""Experimental automatic *candidate* IPF surface labels at knee-flexion minima.

MediaPipe 33-point pose landmarks identify JOINT CENTRES, not the thigh's
visible upper boundary. This module obtains a person-segmentation mask using
PoseLandmarker IMAGE mode on the detected reference frame and looks for the
upward-facing edge of the projected thigh. Surface proposals are unverified.
Nothing here certifies IPF squat depth or overwrites manual annotations.
"""
from __future__ import annotations

from dataclasses import replace
from math import hypot, isfinite
from pathlib import Path

import cv2
from squatvision.video import open_video
import numpy as np

from .analysis import FrameMetrics
from .ipf_depth import AUTOMATED


def minimum_knee_reference(frames: list[FrameMetrics], attempt):
    """Minimum measured knee angle from descent start through ascent start.

    Descent end is not presumed to be the knee-angle minimum; its timestamp
    only resolves exact ties. Unobserved frames cannot produce a measurement.
    """
    events = {e.name: e.time_s for e in attempt.events}
    start = events.get('descent_start', attempt.start_s)
    end = events.get('ascent_start', attempt.end_s)
    if end < start:
        return None
    choices = [f for f in frames if start <= f.time_s <= end
               and f.knee_deg is not None and isfinite(f.knee_deg)
               and 0 < f.knee_deg < 180 and f.points is not None]
    if not choices:
        return None
    approximate_bottom = events.get('descent_end', (start + end) / 2)
    return min(choices, key=lambda f: (f.knee_deg, abs(f.time_s-approximate_bottom), f.frame))


def _visible_top_edge(mask: np.ndarray, centre: np.ndarray,
                      direction_up: np.ndarray, limit: float):
    """Intersect person-mask silhouette along upward thigh normal, in pixels.

    Only returns an edge when the limb centre is foreground and background is
    actually found, so a continuous pelvis/torso region is not guessed as a thigh.
    """
    h, w = mask.shape
    x, y = centre
    if not (0 <= x < w and 0 <= y < h):
        return None
    if mask[min(h-1, max(0, round(y))), min(w-1, max(0, round(x)))] < .72:
        return None
    last = None
    # Bilinear samples reduce jumpiness at diagonally sloping boundaries.
    for u in np.arange(1., limit, .75):
        p = centre + u * direction_up
        xx, yy = p
        if not (1 <= xx < w - 1 and 1 <= yy < h - 1):
            return None
        sampled = float(cv2.getRectSubPix(mask, (1, 1),
                                          (float(xx), float(yy)))[0, 0])
        if sampled < .42:
            if last is None:
                return None
            return last
        if sampled >= .52:
            last = p.copy()
    return None


def estimate_surface_candidates(segmentation: np.ndarray,
                                points: tuple[tuple[float, float], ...],
                                width: int, height: int):
    """Infer provisional upper thigh / top knee pixels from person silhouette.

    Extrapolates the visible superior femoral contour toward joint centres.
    This is a geometric proxy, NOT a recognised anatomical hip crease.
    Returns (hip_xy, knee_xy, heuristic_quality, reason). Failure stays explicit.
    """
    if len(points) < 3:
        return None, None, None, 'missing_pose_joint_centres'
    mask = np.asarray(segmentation, dtype=np.float32)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.shape != (height, width) or not np.isfinite(mask).all():
        return None, None, None, 'invalid_segmentation_mask'
    hip = np.asarray((points[1][0] * width, points[1][1] * height), dtype=float)
    knee = np.asarray((points[2][0] * width, points[2][1] * height), dtype=float)
    v = knee - hip
    length = float(np.linalg.norm(v))
    if not 28 <= length <= hypot(width, height):
        return None, None, None, 'thigh_length_invalid'
    # An overhead/near-vertical thigh cannot give an interpretable upper edge.
    if abs(v[0]) / length < .20:
        return None, None, None, 'thigh_too_vertical_for_silhouette'
    normal = np.array((-v[1], v[0]), dtype=float) / length
    if normal[1] > 0:
        normal = -normal
    # Sampling away from the hip avoids the torso; short 3-sample local line
    # extrapolation aims at the proximal thigh without claiming a skin fold.
    samples = []
    for t in (.19, .28, .37, .73, .82, .90):
        pt = hip + t * v
        edge = _visible_top_edge(mask, pt, normal, min(0.58*length, 90.))
        if edge is None:
            return None, None, None, 'superior_thigh_edge_occluded_or_missing'
        samples.append(edge)
    samples = np.asarray(samples)
    tvals = np.asarray((.19, .28, .37, .73, .82, .90))
    # Fit proximal and distal boundary separately, then modestly extrapolate.
    result = []
    errors = []
    for start, stop, target in ((0, 3, .08), (3, 6, .98)):
        observed = samples[start:stop]
        if np.ptp(observed[:, 0]) > 1.2*length or np.ptp(observed[:, 1]) > 1.2*length:
            return None, None, None, 'incoherent_thigh_contour'
        for axis in (0, 1):
            # Fit in parametric femur coordinates, not raw x to support orientations.
            fit = np.polyfit(tvals[start:stop], observed[:, axis], 1)
            if axis == 0:
                candidate_x = float(np.polyval(fit, target))
            else:
                candidate_y = float(np.polyval(fit, target))
            errors.extend(np.abs(np.polyval(fit, tvals[start:stop]) - observed[:, axis]))
        result.append((candidate_x, candidate_y))
    if any(not (0 <= p[0] < width and 0 <= p[1] < height) for p in result):
        return None, None, None, 'extrapolated_surface_outside_image'
    residual = float(np.max(errors))
    if residual > max(5., length*.09):
        return None, None, None, 'inconsistent_thigh_surface_contour'
    # Quality is a conservative geometric heuristic, not calibrated probability.
    quality = max(.5, min(.85, .75 - residual / max(1., length) * 2.))
    return result[0], result[1], quality, ''


def bottom_neighbourhood(frames, attempt, reference, window_s=.5):
    """Bound candidate review by the detected attempt and available samples."""
    if not isfinite(window_s) or window_s <= 0:
        raise ValueError('Bottom window must be positive and finite')
    if reference is None:
        return []
    return [f for f in frames if max(attempt.start_s, reference.time_s-window_s)
            <= f.time_s <= min(attempt.end_s, reference.time_s+window_s)]


def label_knee_minima(video: Path, frames: list[FrameMetrics], analysis,
                      model: Path, mp, width: int, height: int,
                      window_s: float = .5, diagnostics_dir: Path | None = None):
    """Evaluate all available bottom-window frames; keep original pose analysis.

    Independent proposals and rejection evidence are exported for review.
    Only temporally supported pairs enter numerical measurements. Existing
    manual/flow measurements are retained without promotion of provenance.
    """
    from time import perf_counter
    from .surface_evidence import (
        propose_surfaces, filter_temporal, diagnostic_image, pose_mask_support,
        preferred_review_frame, review_closeups)
    from .geometry import JOINTS
    from .ipf_depth import MANUAL, VERIFIED
    if not analysis.attempts:
        return frames, []
    index = {f.frame: i for i, f in enumerate(frames)}
    updated = list(frames)
    records = []
    cap = open_video(str(video))
    landmarker = None
    init_error = None
    if not cap.isOpened():
        init_error = 'CANNOT_OPEN_SOURCE_VIDEO'
    try:
        if not init_error:
            try:
                options = mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_poses=1, output_segmentation_masks=True)
                landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
            except (ValueError, RuntimeError, AttributeError, TypeError, cv2.error) as exc:
                init_error = f'SEGMENTATION_INITIALISATION_FAILED: {exc}'
        for att in analysis.attempts:
            ref = minimum_knee_reference(frames, att)
            events = {e.name for e in att.events}
            record = {'rep_number': att.number, 'frame': ref.frame if ref else None,
                      'timestamp_s': ref.time_s if ref else None,
                      'minimum_knee_angle_deg': ref.knee_deg if ref else None,
                      'status': 'UNAVAILABLE', 'measurement_source': AUTOMATED,
                      'reason': '', 'window_s': window_s,
                      'phase_bounds_complete': {'descent_start', 'ascent_start'} <= events,
                      'candidates': []}
            region = bottom_neighbourhood(frames, att, ref, window_s)
            # At most this repetition's small review window is kept in memory.
            images = {}
            for f in region:
                row = {'frame': f.frame, 'timestamp_s': f.time_s,
                       'knee_angle_deg': f.knee_deg, 'status': 'UNAVAILABLE',
                       'measurement_source': AUTOMATED, 'margin_px': None, 'reason': ''}
                if f.crease_point is not None and f.knee_top_point is not None:
                    source = f.measurement_source or f.depth_tracking_method
                    row.update(status='MANUAL_VERIFIED' if source in (MANUAL, VERIFIED)
                               else 'EXISTING_UNVERIFIED', measurement_source=source,
                               margin_px=(f.crease_point[1]-f.knee_top_point[1])*height,
                               hip={'point': [f.crease_point[0]*width, f.crease_point[1]*height]},
                               knee={'point': [f.knee_top_point[0]*width, f.knee_top_point[1]*height]},
                               reason='existing_measurement_preserved')
                elif init_error:
                    row['reason'] = init_error
                elif f.points is None:
                    row['reason'] = 'MISSING_POSE_JOINT_CENTRES'
                else:
                    started = perf_counter()
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, f.frame)
                        ok, raw = cap.read()
                        if not ok:
                            raise ValueError('CANNOT_DECODE_SOURCE_FRAME')
                        # Restrict re-detection to the already tracked athlete.
                        # The wide padding keeps the queried normal rays away
                        # from artificial crop borders and includes head/feet.
                        x0,y0,x1,y1 = 0,0,width,height
                        if f.side in JOINTS:
                            joints = np.asarray(f.points)*[width,height]
                            length = np.linalg.norm(joints[2]-joints[1])
                            pad = max(40.,1.25*length)
                            x0=max(0,int(np.floor(joints[:,0].min()-pad)))
                            x1=min(width,int(np.ceil(joints[:,0].max()+pad)))
                            y0=max(0,int(np.floor(joints[:,1].min()-pad)))
                            y1=min(height,int(np.ceil(joints[:,1].max()+pad)))
                        crop=raw[y0:y1,x0:x1]
                        if crop.size == 0:
                            raise ValueError('EMPTY_SEGMENTATION_ROI')

                        # Feed MediaPipe an aligned RGB image. Some MediaPipe
                        # 0.10.x float32 masks have a non-contiguous row stride
                        # when the width is not divisible by four.
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        roi_h, roi_w = rgb.shape[:2]
                        pad_right = (-roi_w) % 4
                        if pad_right:
                            rgb = cv2.copyMakeBorder(
                                rgb, 0, 0, 0, pad_right,
                                cv2.BORDER_REPLICATE)

                        img = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb))
                        # This inference must run before reading `detected`.
                        detected = landmarker.detect(img)
                        row['pose_roi_px'] = [x0, y0, x1, y1]
                        if not detected.segmentation_masks:
                            images[f.frame] = (raw, None, f.points)
                            raise ValueError('SEGMENTATION_MASK_UNAVAILABLE')

                        mask_image = detected.segmentation_masks[0]
                        # Calling numpy_view() on a non-contiguous float32 mask
                        # can terminate the Python process inside MediaPipe C++.
                        # Never attempt conversion if the layout is unsafe.
                        if not mask_image.is_contiguous():
                            raise ValueError('SEGMENTATION_MASK_NONCONTIGUOUS')
                        padded_mask = mask_image.numpy_view()

                        # MediaPipe may return either (H, W) or (H, W, 1).
                        # Remove only a trailing singleton channel dimension.
                        if padded_mask.ndim == 3 and padded_mask.shape[-1] == 1:
                            padded_mask = padded_mask[..., 0]

                        expected_shape = (roi_h, roi_w + pad_right)

                        if padded_mask.shape != expected_shape:
                            raise ValueError(
                                f"SEGMENTATION_ROI_SHAPE_MISMATCH: "
                                f"got {padded_mask.shape}, expected {expected_shape}"
                            )

                        # Remove artificial right-side padding and restore
                        # the original full-resolution video coordinates.
                        local_mask = np.ascontiguousarray(
                            padded_mask[:, :roi_w], dtype=np.float32)
                        mask = np.zeros((height, width), dtype=np.float32)
                        mask[y0:y1, x0:x1] = local_mask
                        images[f.frame] = (raw, mask, f.points)
                        # Segmentation is accepted only when the ORIGINAL
                        # VIDEO-mode pose's hip, knee and thigh centreline are
                        # supported by one connected foreground component.
                        # This is evidence of association, not anatomical proof.
                        association = pose_mask_support(mask, f.points, width, height)
                        row['original_pose_mask_support'] = association
                        if not association['supported']:
                            raise ValueError(association['reason'])

                        # IMAGE-mode pose is unstable when crop framing changes.
                        # Keep its displacement for diagnostics. Moderate shifts
                        # do not automatically invalidate an otherwise supported
                        # person mask, but extreme shifts remain a safety rejection.
                        poses = getattr(detected, 'pose_landmarks', None)
                        if poses and f.side in JOINTS:
                            ids = JOINTS[f.side]
                            pose_xy = (np.array([(poses[0][i].x, poses[0][i].y)
                                                 for i in ids], dtype=float)
                                       * [roi_w + pad_right, roi_h] + [x0, y0])
                            old = np.asarray(f.points, dtype=float) * [width, height]
                            distances = np.linalg.norm(pose_xy - old, axis=1)
                            if not np.isfinite(distances).all():
                                raise ValueError('SEGMENTATION_POSE_INVALID')
                            body_span = max(1., float(np.linalg.norm(np.ptp(old, axis=0))))
                            row['redetection_joint_displacements_px'] = distances.tolist()
                            # Crop-sensitive differences are a warning, not
                            # sufficient proof of a wrong athlete.
                            if (np.median(distances) > .15 * body_span or
                                    np.max(distances) > .30 * body_span):
                                row['second_pass_pose_warning'] = 'CROP_SENSITIVE_JOINT_DISAGREEMENT'
                            # Very large displacements indicate that the second
                            # pass may have latched onto a different person.
                            if (np.median(distances) > max(65., .50 * body_span) or
                                    np.max(distances) > max(85., .70 * body_span)):
                                raise ValueError('SECOND_PASS_PERSON_IDENTITY_UNCERTAIN')
                        elif not poses:
                            row['second_pass_pose_warning'] = 'NO_SECOND_PASS_POSE'

                        row.update(propose_surfaces(raw, mask, f.points, width, height))
                    except (ValueError, RuntimeError, AttributeError, TypeError, IndexError, cv2.error) as exc:
                        row['reason'] = str(exc)
                    row['processing_ms'] = (perf_counter()-started)*1000
                record['candidates'].append(row)
            rows = filter_temporal(record['candidates'])
            suggested, why = preferred_review_frame(rows, ref.frame if ref else -1)
            record['suggested_review_frame'] = suggested
            record['suggested_review_reason'] = why
            for row in rows:
                f = frames[index[row['frame']]]
                if row['status'] == 'AUTOMATIC_UNVERIFIED':
                    hp, kp = row['hip']['point'], row['knee']['point']
                    updated[index[f.frame]] = replace(
                        f, crease_point=(hp[0]/width, hp[1]/height),
                        knee_top_point=(kp[0]/width, kp[1]/height),
                        crease_to_knee_top_delta=(hp[1]-kp[1])/height,
                        hip_crease_tracking_confidence=row['hip']['quality'],
                        knee_top_tracking_confidence=row['knee']['quality'],
                        depth_tracking_method='auto_surface_multiframe',
                        measurement_source=AUTOMATED, depth_unavailable_reason='')
                elif row['status'] == 'UNAVAILABLE' and f.crease_point is None and f.depth_tracking_method == 'not_calibrated':
                    updated[index[f.frame]] = replace(f, depth_unavailable_reason=row['reason'],
                                                     depth_tracking_method='auto_surface_multiframe')
            valid = [r for r in rows if r['status'] in
                     ('AUTOMATIC_UNVERIFIED', 'MANUAL_VERIFIED', 'EXISTING_UNVERIFIED')
                     and r.get('margin_px') is not None]
            reference_row = next((r for r in rows if ref and r['frame'] == ref.frame), None)
            record['valid_proposal_frames'] = sum(r['status'] == 'AUTOMATIC_UNVERIFIED' for r in rows)
            record['evaluated_frames'] = len(rows)
            record['maximum_observed'] = max(valid, key=lambda r: r['margin_px']).copy() if valid else None
            record['review_frame'] = ref.frame if ref else None
            if reference_row:
                record['reference_status'] = reference_row['status']
                record['status'] = ('EXISTING_SURFACE_POINTS' if reference_row['status'] in
                                    ('MANUAL_VERIFIED', 'EXISTING_UNVERIFIED') else
                                    'UNVERIFIED_SURFACE_PROPOSAL' if valid else 'UNAVAILABLE')
                record['reason'] = reference_row['reason']
                record['measurement_source'] = reference_row['measurement_source']
                # Backward-compatible coordinates refer ONLY to reference frame.
                for part, key in [('hip','hip_surface_px'), ('knee','knee_top_px')]:
                    record[key] = reference_row.get(part, {}).get('point')
                record['margin_px'] = reference_row.get('margin_px')
            else:
                record['reason'] = 'NO_VALID_KNEE_REFERENCE'
            if diagnostics_dir is not None:
                diagnostics_dir.mkdir(parents=True, exist_ok=True)
                for row in rows:
                    if row['frame'] not in images:
                        continue
                    raw, mask, points = images[row['frame']]
                    path = diagnostics_dir/f'rep_{att.number:02d}_frame_{row["frame"]:06d}.jpg'
                    if not cv2.imwrite(str(path), diagnostic_image(raw, mask, points, dict(row, rep_number=att.number))):
                        raise RuntimeError(f'Cannot write diagnostic image: {path}')
                    row['diagnostic_image'] = str(Path(diagnostics_dir.name)/path.name)
                    # Export high-magnification source/segmentation regions,
                    # including partial/rejected evidence. These are review
                    # aids only and never create anatomical measurements.
                    if points is not None:
                        closeup = diagnostics_dir/f'rep_{att.number:02d}_frame_{row["frame"]:06d}_closeups.jpg'
                        if not cv2.imwrite(str(closeup), review_closeups(raw, mask, points, row)):
                            raise RuntimeError(f'Cannot write review closeups: {closeup}')
                        row['closeups_image'] = str(Path(diagnostics_dir.name)/closeup.name)
                record['diagnostic_image'] = reference_row.get('diagnostic_image') if reference_row else None
            records.append(record)
    finally:
        if landmarker is not None:
            landmarker.close()
        cap.release()
    return updated, records
