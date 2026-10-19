"""Experimental image-plane IPF surface-depth assessments.

The upper-thigh-at-hip and top-of-knee are independently selected visual
SURFACES, *not* MediaPipe joint centres. Pixel margins are not centimetres.
Uncertainty is a conservative user-configured engineering bound, not a
statistical confidence interval or competition-validated error distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot, isfinite
from pathlib import Path
from typing import Optional
import json

from .analysis import FrameMetrics

AUTOMATED = 'AUTOMATED_TRACKING'
MANUAL = 'MANUAL_ANNOTATION'
VERIFIED = 'MANUAL_VERIFIED_TRACKING'
UNAVAILABLE = 'UNAVAILABLE'
ABOVE = 'ESTIMATED_ABOVE_DEPTH_THRESHOLD'
BELOW = 'ESTIMATED_BELOW_DEPTH_THRESHOLD'
UNCERTAIN = 'BORDERLINE_UNCERTAIN'


@dataclass(frozen=True)
class ManualPoint:
    frame: int
    upper_thigh_at_hip: tuple[float, float]  # source image pixels
    top_of_knee: tuple[float, float]
    uncertainty_px: float = 6.0
    source: str = MANUAL
    note: str = ''


@dataclass(frozen=True)
class Measurement:
    frame: int
    timestamp_s: float
    upper_thigh_at_hip_x: Optional[float]
    upper_thigh_at_hip_y: Optional[float]
    knee_top_x: Optional[float]
    knee_top_y: Optional[float]
    depth_margin_px: Optional[float]
    depth_margin_normalized: Optional[float]
    depth_status: str
    depth_uncertainty_px: Optional[float]
    measurement_source: str
    tracking_quality: Optional[float]
    unavailable_reason: str

    def to_dict(self) -> dict:
        return vars(self).copy()


def validate_manual_point(point: ManualPoint, width: int, height: int) -> None:
    if point.frame < 0 or int(point.frame) != point.frame:
        raise ValueError('Manual annotation frame must be a nonnegative integer')
    if point.source not in (MANUAL, VERIFIED):
        raise ValueError('Invalid manual measurement provenance')
    if not (isfinite(point.uncertainty_px) and point.uncertainty_px > 0):
        raise ValueError('Manual uncertainty must be positive and finite')
    for name,p in [('upper thigh',point.upper_thigh_at_hip),('knee top',point.top_of_knee)]:
        if len(p)!=2 or not all(isfinite(float(q)) for q in p):
            raise ValueError(f'Invalid {name} coordinates')
        if not (0 <= p[0] < width and 0 <= p[1] < height):
            raise ValueError(f'{name} lies outside video frame')
    # Not using pose joint centres or angle heuristics to validate anatomy.


def save_manual(path: Path, video_name: str, width: int, height: int, fps: float,
                points: list[ManualPoint]) -> None:
    if width <= 0 or height <= 0 or not isfinite(fps) or fps <= 0:
        raise ValueError('Invalid video metadata')
    seen=set()
    for p in points:
        validate_manual_point(p,width,height)
        if p.frame in seen:
            raise ValueError('Duplicate annotated frame; edit/replace rather than append')
        seen.add(p.frame)
    payload={'version':1,'video_name':video_name,'width':width,'height':height,'fps':fps,
             'annotations':[{**vars(p),'upper_thigh_at_hip':list(p.upper_thigh_at_hip),
                              'top_of_knee':list(p.top_of_knee)} for p in sorted(points,key=lambda a:a.frame)]}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2),encoding='utf-8')


def load_manual(path: Path, video_name: str, width: int, height: int, fps: float) -> dict[int,ManualPoint]:
    data=json.loads(path.read_text(encoding='utf-8'))
    if (data.get('version')!=1 or data.get('video_name')!=video_name or
        data.get('width')!=width or data.get('height')!=height or
        abs(float(data.get('fps',0))-fps)>0.05):
        raise ValueError('Manual annotations do not match video name, resolution or FPS')
    points={}
    for d in data['annotations']:
        point=ManualPoint(frame=d['frame'],upper_thigh_at_hip=tuple(d['upper_thigh_at_hip']),
                          top_of_knee=tuple(d['top_of_knee']),uncertainty_px=d.get('uncertainty_px',6.),
                          source=d.get('source',MANUAL), note=d.get('note',''))
        validate_manual_point(point,width,height)
        if point.frame in points:
            raise ValueError('Duplicate annotation frame')
        points[point.frame]=point
    return points


def apply_manual(frame: FrameMetrics, point: ManualPoint, width: int, height: int) -> FrameMetrics:
    """Manual depth works even if MediaPipe landmarks/flow are completely missing."""
    validate_manual_point(point,width,height)
    if point.frame!=frame.frame:
        raise ValueError('Annotation frame does not match video frame')
    hp=point.upper_thigh_at_hip
    kp=point.top_of_knee
    return replace(frame,crease_point=(hp[0]/width,hp[1]/height),
                   knee_top_point=(kp[0]/width,kp[1]/height),
                   crease_to_knee_top_delta=(hp[1]-kp[1])/height,
                   hip_crease_tracking_confidence=None,knee_top_tracking_confidence=None,
                   depth_tracking_method=point.source,depth_unavailable_reason='',
                   manual_uncertainty_px=point.uncertainty_px,
                   measurement_source=point.source)


def projected_thigh_length(frame: FrameMetrics, width: int, height: int) -> Optional[float]:
    if not frame.points or len(frame.points)<3:
        return None
    hip,knee=frame.points[1:3]
    value=hypot((hip[0]-knee[0])*width,(hip[1]-knee[1])*height)
    return value if value>1 and isfinite(value) else None


def measure(frame: FrameMetrics, width: int, height: int,
            automated_uncertainty_px: float=8.0, manual_default_px: float=6.0,
            require_manual_verification: bool = False) -> Measurement:
    source=frame.measurement_source if frame.measurement_source!='' else (
        frame.depth_tracking_method if frame.depth_tracking_method in (MANUAL,VERIFIED)
        else AUTOMATED if frame.depth_tracking_method=='local_optical_flow' else 'NONE')
    conf = (min(frame.hip_crease_tracking_confidence,frame.knee_top_tracking_confidence)
            if frame.hip_crease_tracking_confidence is not None and
               frame.knee_top_tracking_confidence is not None else None)
    if (frame.crease_point is None or frame.knee_top_point is None or
        (source==AUTOMATED and (conf is None or conf<.5))):
        return Measurement(frame.frame,frame.time_s,None,None,None,None,None,None,
                           UNAVAILABLE,None,source,conf,
                           frame.depth_unavailable_reason or 'no_reliable_anatomical_surface_points')
    hx,hy=(frame.crease_point[0]*width,frame.crease_point[1]*height)
    kx,ky=(frame.knee_top_point[0]*width,frame.knee_top_point[1]*height)
    if not all(isfinite(v) for v in (hx,hy,kx,ky)):
        return Measurement(frame.frame,frame.time_s,None,None,None,None,None,None,
                           UNAVAILABLE,None,source,conf,'nonfinite_coordinate')
    delta=hy-ky
    length=projected_thigh_length(frame,width,height)
    normalised=delta/length if length else None
    # Engineering error budget only. Includes manual marking ambiguity and
    # tracking instability. User must validate these margins on real footage.
    if source in (MANUAL,VERIFIED):
        uncertainty=frame.manual_uncertainty_px if frame.manual_uncertainty_px is not None else manual_default_px
    else:
        uncertainty=automated_uncertainty_px+max(0.,1.-(conf or 0.))*6.0
    status=BELOW if delta>uncertainty else ABOVE if delta < -uncertainty else UNCERTAIN
    # Optical-flow consistency is NOT anatomical correctness.  When strict
    # reporting is enabled, retain its raw estimate for diagnostics but never
    # promote it to an IPF above/below decision without an independently marked
    # source frame. This is deliberately conservative for gym footage.
    reason = ''
    if require_manual_verification and source == AUTOMATED:
        status = UNCERTAIN
        reason = 'automated_surface_locations_not_manually_verified'
    return Measurement(frame.frame,frame.time_s,hx,hy,kx,ky,delta,normalised,status,
                       uncertainty,source,conf,reason)


def attempt_depth(rows: list[Measurement], start_s: float, end_s: float,
                  number: int, completed: bool, bottom_time_s: float | None = None) -> dict:
    """Summarise the *observed* maximum, never claim depth not reached across gaps.

    Depth at a missed bottom is unknown. Optical flow alone is provisional.
    A valid manual BELOW observation supports that an observed image-plane
    threshold was crossed. ABOVE across a whole rep requires review of the
    entire relevant bottom region, which this sampler cannot guarantee.
    """
    in_range = [r for r in rows if start_s <= r.timestamp_s <= end_s]
    valid = [r for r in in_range if r.depth_margin_px is not None]
    if not valid:
        return {'rep_number':number,'completed':completed,
                'depth_assessment':{'status':UNAVAILABLE,'measurement_source':'NONE',
                                    'maximum_depth_frame':None,'maximum_depth_timestamp_s':None,
                                    'depth_margin_px':None,'depth_margin_normalized':None,
                                    'uncertainty_px':None,'valid_frames':0,
                                    'unavailable_frames':len(in_range),
                                    'interpretation':'No reliable anatomical-surface measurements in this attempt'}}
    peak = max(valid,key=lambda r:r.depth_margin_px)
    manual = [r for r in valid if r.measurement_source in (MANUAL, VERIFIED)]
    manual_below = [r for r in manual if r.depth_status == BELOW]
    # Any manually marked below-threshold observation is evidence that the
    # *image-plane* positions crossed the line. Optical flow is not that proof.
    if manual_below:
        peak = max(manual_below,key=lambda r:r.depth_margin_px)
        status = BELOW
        interpretation = ('At least one manually annotated frame shows the estimated hip surface below '
                          'the knee-top surface by more than the selected uncertainty tolerance. '
                          'This is not an official IPF decision.')
    else:
        # A negative measurement from an early frame cannot establish whether
        # the bottom was deep enough. Never convert gaps into an ABOVE verdict.
        status = UNCERTAIN
        if not manual:
            interpretation = ('Only unverified automatic surface estimates exist. Anatomical locations must '
                              'be reviewed on original footage near the squat bottom; no depth verdict.')
        elif bottom_time_s is not None and not any(abs(r.timestamp_s-bottom_time_s) <= .35 for r in manual):
            interpretation = ('Manual annotations do not cover the estimated squat bottom. '
                              'Review frames near that position; no overall depth verdict.')
        else:
            interpretation = ('Observed manual measurements do not establish that sufficient depth was '
                              'reached; unsampled deeper frames remain possible. No definitive above-depth verdict.')
    # Keep unavailable count separate from provisional observations.
    return {'rep_number':number,'completed':completed,'depth_assessment':{
        'status':status,'measurement_source':peak.measurement_source,
        'maximum_depth_frame':peak.frame,'maximum_depth_timestamp_s':round(peak.timestamp_s,3),
        'depth_margin_px':round(peak.depth_margin_px,2),
        'depth_margin_normalized':round(peak.depth_margin_normalized,5)
             if peak.depth_margin_normalized is not None else None,
        'uncertainty_px':round(peak.depth_uncertainty_px,2),
        'upper_thigh_at_hip_px':[peak.upper_thigh_at_hip_x,peak.upper_thigh_at_hip_y],
        'top_of_knee_px':[peak.knee_top_x,peak.knee_top_y],
        'tracking_quality':peak.tracking_quality,'valid_frames':len(valid),
        'unavailable_frames':len(in_range)-len(valid),
        'manually_annotated_frames':len(manual),
        'estimated_bottom_timestamp_s':bottom_time_s,
        'interpretation':'Maximum of observed frames only. '+interpretation}}
