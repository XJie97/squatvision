"""Pose-constrained visible boundary proposals; never semantic anatomy labels.

The mask supplies foreground support. RGB gradients can reveal internal
boundaries hidden by the whole-person silhouette, but can also be clothing.
Every accepted result remains unverified, including temporally stable ones.
"""
from __future__ import annotations

import cv2
import numpy as np


def _sample(image, xy):
    return cv2.remap(image, xy[..., 0].astype(np.float32),
                     xy[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _region(mask, lab, hip, axis, normal, length, target):
    """Observe a boundary AT the target with local support, no extrapolation.

    Five parallel rays span 16% of femur length. All five must support a
    coherent edge; two competing boundary bands are rejected. The centre ray
    gives the coordinate. A silhouette edge takes priority over internal RGB.
    """
    h, w = mask.shape
    ts = target + np.linspace(-.08, .08, 5)
    us = np.arange(2., min(.58 * length, 90.), .75)
    failure = {'point': None, 'method': None, 'quality': None,
               'reason': 'SURFACE_OCCLUDED_OR_MISSING', 'contour_px': []}
    if len(us) < 6:
        return {**failure, 'reason': 'ROI_TOO_SMALL'}
    xy = hip + ts[:, None, None] * axis + us[None, :, None] * normal
    if np.any(xy[..., 0] < 1) or np.any(xy[..., 0] >= w-1) or \
            np.any(xy[..., 1] < 1) or np.any(xy[..., 1] >= h-1):
        return {**failure, 'reason': 'ROI_OUTSIDE_IMAGE'}
    centres = hip + ts[:, None] * axis
    if np.any(_sample(mask, centres[:, None, :]) < .72):
        return {**failure, 'reason': 'POSE_MASK_MISMATCH'}
    profiles = _sample(mask, xy)
    offsets = []
    for profile in profiles:
        bg = np.flatnonzero(profile < .42)
        if not len(bg) or bg[0] < 2:
            offsets = []
            break
        offsets.append(float(us[bg[0]-1]))
    method = 'person_silhouette'
    if not offsets:
        if lab is None:
            return failure
        colours = _sample(lab, xy)
        # A local normal colour difference over 3 px, independent of RGB sign.
        strength = np.linalg.norm(colours[:, 4:] - colours[:, :-4], axis=2)
        candidates = []
        for i, row in enumerate(strength):
            # Internal edges must have foreground on both sides. Do not allow
            # the background/rack to become an internal thigh boundary.
            valid = (profiles[i, 4:] > .72) & (profiles[i, :-4] > .72)
            row = np.where(valid, row, 0.)
            peaks = [j for j in range(1, len(row)-1)
                     if row[j] >= 16. and row[j] >= row[j-1] and row[j] > row[j+1]]
            if not peaks:
                return {**failure, 'reason': 'NO_SUPPORTED_INTERNAL_BOUNDARY'}
            peaks.sort(key=lambda j: row[j], reverse=True)
            best = peaks[0]
            if any(abs(us[j+2]-us[best+2]) > 4. and row[j] >= .7*row[best]
                   for j in peaks[1:]):
                return {**failure, 'reason': 'ANATOMICAL_BOUNDARY_AMBIGUOUS'}
            candidates.append(float(us[best+2]))
        offsets = candidates
        method = 'internal_rgb_boundary'
    offsets = np.asarray(offsets)
    fit = np.polyfit(ts-target, offsets, 1)
    residual = float(np.max(np.abs(np.polyval(fit, ts-target)-offsets)))
    boundary_shape = 'linear'
    if residual > max(2.5, .035*length) or abs(fit[0])/length > .8:
        # A knee silhouette can be smoothly curved. A straight-line fit can
        # reject a genuine knee cap. Permit a tightly constrained quadratic
        # ONLY for the visible distal silhouette, never for RGB/clothing edges
        # or the obscured proximal hip. The returned point stays an observed
        # centre-ray intersection, not the fitted/extrapolated coordinate.
        if target < .95 or method != 'person_silhouette':
            return {**failure, 'reason': 'INCOHERENT_LOCAL_BOUNDARY'}
        curve = np.polyfit(ts-target, offsets, 2)
        curve_residual = float(np.max(np.abs(np.polyval(curve, ts-target)-offsets)))
        curve_excursion = float(np.ptp(offsets))
        max_adjacent_jump = float(np.max(np.abs(np.diff(offsets))))
        if (curve_residual > 1.25 or curve_excursion > .16*length or
                max_adjacent_jump > max(3., .065*length)):
            return {**failure, 'reason': 'INCOHERENT_LOCAL_BOUNDARY'}
        residual = curve_residual
        boundary_shape = 'smooth_knee_curve'
    contour = centres + offsets[:, None]*normal
    point = contour[2]  # observed central-ray edge; no extrapolated coordinates
    return {'point': point.tolist(), 'method': method,
            'quality': float(max(.5, (.75 if method == 'person_silhouette' else .6)
                                 - residual / length)),
            'reason': '', 'contour_px': contour.tolist(),
            'fit_residual_px': residual, 'boundary_shape': boundary_shape,
            'offset_fraction': float(offsets[2]/length)}



def pose_mask_support(segmentation, points, width, height):
    """Check whether a mask supports the FIRST-PASS athlete's thigh.

    The second IMAGE-mode pose result is not a stable identity anchor for a
    cropped frame. Require the original hip, knee, and most of the connecting
    femoral centreline to lie in a *single* foreground component instead.
    This only associates a mask with a pose; it does NOT identify anatomy.
    """
    failure = {'supported': False, 'reason': 'POSE_MASK_MISMATCH'}
    mask = np.asarray(segmentation, dtype=np.float32)
    if mask.shape == (height, width, 1):
        mask = mask[..., 0]
    if mask.shape != (height, width) or not np.isfinite(mask).all():
        return {**failure, 'reason': 'INVALID_SEGMENTATION_MASK'}
    if points is None or len(points) < 3:
        return {**failure, 'reason': 'MISSING_POSE_JOINT_CENTRES'}
    coords = np.asarray(points[1:3], dtype=float)
    if coords.shape != (2, 2) or not np.isfinite(coords).all():
        return {**failure, 'reason': 'INVALID_POSE_JOINT_CENTRES'}
    coords *= [width, height]
    if np.any(coords < 0) or np.any(coords >= [width, height]):
        return {**failure, 'reason': 'POSE_OUTSIDE_IMAGE'}

    _, components = cv2.connectedComponents((mask >= .52).astype(np.uint8))

    def component_at(pt):
        x = min(width - 1, max(0, int(round(pt[0]))))
        y = min(height - 1, max(0, int(round(pt[1]))))
        return int(components[y, x])

    hip_component, knee_component = (component_at(coords[0]), component_at(coords[1]))
    if hip_component == 0 or hip_component != knee_component:
        return {**failure, 'hip_component': hip_component,
                'knee_component': knee_component}
    middle = [component_at(coords[0] + t * (coords[1] - coords[0]))
              for t in (.25, .50, .75)]
    supported_middle = sum(v == hip_component for v in middle)
    result = {'supported': supported_middle >= 2,
              'reason': '' if supported_middle >= 2 else 'THIGH_CENTERLINE_NOT_IN_MASK',
              'hip_component': hip_component,
              'knee_component': knee_component,
              'supported_middle_samples': supported_middle,
              'total_middle_samples': 3}
    return result


def propose_surfaces(image, segmentation, points, width, height):
    """Return independent surface evidence; missing hip does not erase knee."""
    empty = lambda reason: {'point': None, 'method': None, 'quality': None,
                            'reason': reason, 'contour_px': []}
    def fail(reason):
        return {'hip': empty(reason), 'knee': empty(reason), 'reason': reason,
                'status': 'UNAVAILABLE', 'margin_px': None}
    if points is None or len(points) < 3:
        return fail('MISSING_POSE_JOINT_CENTRES')
    mask = np.asarray(segmentation, dtype=np.float32)
    if mask.shape == (height, width, 1):
        mask = mask[..., 0]
    if mask.shape != (height, width) or not np.isfinite(mask).all() or \
            np.min(mask) < 0 or np.max(mask) > 1:
        return fail('INVALID_SEGMENTATION_MASK')
    coords = np.asarray(points[1:3], dtype=float)*[width, height]
    if not np.isfinite(coords).all():
        return fail('INVALID_POSE_JOINT_CENTRES')
    hip, knee = coords
    if np.any(coords < 0) or np.any(coords >= [width, height]):
        return fail('POSE_OUTSIDE_IMAGE')
    axis = knee-hip
    length = float(np.linalg.norm(axis))
    if length < 28:
        return fail('PROJECTED_THIGH_TOO_SHORT')
    if abs(axis[0])/length < .2:
        return fail('THIGH_TOO_VERTICAL')
    # Anchor the segmentation to the original (VIDEO-mode) pose, rather than
    # trusting the independent IMAGE-mode pose to identify the same athlete.
    association = pose_mask_support(mask, points, width, height)
    if not association['supported']:
        return fail(association['reason'])
    normal = np.array([-axis[1], axis[0]])/length
    if normal[1] > 0:
        normal = -normal
    lab = None
    if image is not None:
        if image.shape != (height, width, 3) or image.dtype != np.uint8:
            return fail('INVALID_IMAGE')
        lab = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    hp = _region(mask, lab, hip, axis, normal, length, 0.)
    kp = _region(mask, lab, hip, axis, normal, length, 1.)
    paired = hp['point'] is not None and kp['point'] is not None
    return {'hip': hp, 'knee': kp,
            'reason': '; '.join(f'{name.upper()}: {p["reason"]}'
                                for name, p in [('hip', hp), ('knee', kp)] if p['reason']),
            'status': 'AUTOMATIC_CANDIDATE' if paired else 'UNAVAILABLE',
            'margin_px': hp['point'][1]-kp['point'][1] if paired else None,
            'projected_thigh_length_px': length,
            'anatomical_identity': 'UNVERIFIED; clothing and overlapping people can produce false edges'}


def filter_temporal(rows, max_gap_s=.15):
    """Validate raw paired candidates without inventing missing observations.

    Distinguish an isolated raw image-boundary difference from a supported
    automatic measurement. A rejected raw difference is *never* exported in
    `margin_px`; it is retained under `raw_candidate_margin_px` for review.
    """
    paired = [r for r in rows if r.get('margin_px') is not None and
              r.get('status') == 'AUTOMATIC_CANDIDATE']
    # Snapshot candidates before updating statuses; maintain deterministic
    # peer matching across the entire neighbourhood.
    for row in paired:
        row['raw_candidate_margin_px'] = row['margin_px']
    for row in paired:
        peers = [p for p in paired if p is not row and
                 0 < abs(p['timestamp_s']-row['timestamp_s']) <= max_gap_s]
        good = [p for p in peers if all(
            abs(p[part]['offset_fraction']-row[part]['offset_fraction']) <= .12
            for part in ('hip', 'knee'))]
        if len(good) >= 2:
            row['status'] = 'AUTOMATIC_UNVERIFIED'
        else:
            row['status'] = 'UNAVAILABLE'
            row['margin_px'] = None
            row['reason'] = ('TEMPORAL_INCONSISTENCY' if len(peers) >= 2
                             else 'INSUFFICIENT_VALID_FRAMES')
        row['temporal_support_frames'] = len(good)
    return rows


def preferred_review_frame(rows, reference_frame):
    """Choose an informative *manual-review* frame, never a depth verdict.

    Complete observed pairs are easier for a person to accept or correct;
    partial pairs are more useful than empty images. Stay near the movement
    bottom on ties. Even rejected proposals remain unverified evidence.
    """
    if not rows:
        return None, 'NO_REVIEW_FRAMES'
    def rank(row):
        n = sum(row.get(part, {}).get('point') is not None for part in ('hip', 'knee'))
        status = row.get('status')
        # Manual evidence always takes precedence; automatic consistency next.
        valid = int(status in ('MANUAL_VERIFIED', 'AUTOMATIC_UNVERIFIED'))
        return (valid, n, -abs(row['frame']-reference_frame), -row['frame'])
    chosen = max(rows, key=rank)
    n = sum(chosen.get(part, {}).get('point') is not None for part in ('hip', 'knee'))
    why = ('EXISTING_VERIFIED_POINTS' if chosen.get('status') == 'MANUAL_VERIFIED'
           else 'TWO_UNVERIFIED_POINTS_FOR_HUMAN_REVIEW' if n == 2
           else 'ONE_PARTIAL_POINT_FOR_HUMAN_REVIEW' if n == 1
           else 'NO_SURFACE_POINTS_AVAILABLE')
    return chosen['frame'], why


def diagnostic_image(image, mask, points, row):
    """Separate enlarged image/mask panels with a readable off-image caption."""
    h, w = image.shape[:2]
    marked = image.copy()
    masked = image.copy() if mask is None else cv2.addWeighted(
        image, .45, cv2.applyColorMap((np.clip(mask, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), .55, 0)
    for canvas in (marked, masked):
        if points:
            for p in points[1:3]:
                cv2.drawMarker(canvas, tuple(np.rint(np.array(p)*[w,h]).astype(int)),
                               (0,180,255), cv2.MARKER_CROSS, 11, 1)
        for part, color in [('hip', (70,240,70)), ('knee', (245,150,245))]:
            evidence = row.get(part, {})
            contour = evidence.get('contour_px', [])
            if contour:
                cv2.polylines(canvas, [np.rint(contour).astype(np.int32)], False, color, 1)
            if evidence.get('point') is not None:
                cv2.circle(canvas, tuple(np.rint(evidence['point']).astype(int)), 5, color, 1)
    scale = max(1., min(2., 900/h))
    panels = cv2.resize(np.hstack([marked, masked]), None, fx=scale, fy=scale)
    canvas = np.full((panels.shape[0]+190, max(1000, panels.shape[1]), 3), 22, np.uint8)
    canvas[:panels.shape[0], :panels.shape[1]] = panels
    raw_margin = row.get('raw_candidate_margin_px', row.get('margin_px'))
    margin = f'{raw_margin:+.2f}' if raw_margin is not None else 'unavailable'
    lines = [f"Rep {row.get('rep_number','?')} | Frame {row['frame']} | {row['timestamp_s']:.3f}s | {row.get('status')}",
             'Crosses: joint centres. Circles: visible-boundary proposals, not verified anatomy.',
             'Left: source + selected contour. Right: whole-person mask (NOT body-part labels).',
             f"Hip: {row.get('hip', {}).get('method') or row.get('hip', {}).get('reason', 'unavailable')}",
             f"Knee: {row.get('knee', {}).get('method') or row.get('knee', {}).get('reason', 'unavailable')}",
             f"Raw candidate margin: {margin} px | {row.get('reason', '')}"]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line[:135], (12, panels.shape[0]+25+i*27),
                    cv2.FONT_HERSHEY_SIMPLEX, .47, (240,240,240), 1, cv2.LINE_AA)
    return canvas


def review_closeups(image, mask, points, row):
    """Export inspectable original-pixel hip/knee crops for manual review.

    Four panels: source HIP/KNEE, then foreground-mask HIP/KNEE. Enlargement
    does not recover hidden details or establish anatomical identity. Crops
    are aligned to the ORIGINAL pose, not the crop-sensitive IMAGE-mode pose.
    """
    h, w = image.shape[:2]
    if points is None or len(points) < 3:
        raise ValueError('Pose centres needed for review closeups')
    centres = np.asarray(points[1:3], dtype=float) * [w, h]
    if not np.isfinite(centres).all():
        raise ValueError('Invalid pose centres for review closeups')
    femur = float(np.linalg.norm(centres[1]-centres[0]))
    half = int(np.clip(femur*1.12, 44, 95))
    size = 420
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 3 and m.shape[-1] == 1:
            m = m[..., 0]
        if m.shape != (h, w):
            raise ValueError('Review mask must match source frame')
        heat = cv2.applyColorMap((np.clip(m, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        masked = cv2.addWeighted(image, .45, heat, .55, 0)
    else:
        masked = image.copy()

    result = np.full((size*2+132, size*2, 3), 26, dtype=np.uint8)
    for i, part in enumerate(('hip','knee')):
        centre = centres[i]
        x, y = np.rint(centre).astype(int)
        x0,y0,x1,y1 = x-half,y-half,x+half,y+half
        for j, source in enumerate((image, masked)):
            # Explicit border padding keeps the review radius consistent even
            # when a landmark lies close to the original image boundary.
            patch = cv2.getRectSubPix(cv2.copyMakeBorder(
                source,half,half,half,half,cv2.BORDER_CONSTANT,value=(20,20,20)),
                (half*2,half*2),(float(x+half),float(y+half)))
            zoom = cv2.resize(patch,(size,size),interpolation=cv2.INTER_NEAREST)
            def screen_xy(pt):
                return (int(round((pt[0]-x0)*size/(2*half))),
                        int(round((pt[1]-y0)*size/(2*half))))
            cv2.drawMarker(zoom,screen_xy(centre),(0,180,255),cv2.MARKER_CROSS,18,2)
            proposal = row.get(part,{}).get('point')
            if proposal is not None:
                cv2.circle(zoom,screen_xy(proposal),9,(70,240,70) if i==0 else (245,150,245),2)
            cv2.putText(zoom,part.upper()+(' SOURCE' if j==0 else ' MASK'),(12,30),
                        cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),2,cv2.LINE_AA)
            result[j*size:(j+1)*size,i*size:(i+1)*size]=zoom
    legend = ['ORANGE cross = original hip/knee JOINT centre; circles = UNVERIFIED image edges',
              f"Frame {row.get('frame','?')} | Rejected: {row.get('reason','')}"[:110],
              'Original pixels enlarged; a shirt/shorts seam is NOT known anatomy']
    for k,line in enumerate(legend):
        cv2.putText(result,line,(15,2*size+32+k*32),cv2.FONT_HERSHEY_SIMPLEX,
                    .52,(240,240,240),1,cv2.LINE_AA)
    return result
