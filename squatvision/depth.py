"""Calibrated 2-D hip-crease and knee-top *tracking estimates*.

MediaPipe does NOT expose either anatomical surface point. The user selects them
on a reference frame, then a thigh-attached similarity transform propagates the
selected positions. This is motion estimation, not visual re-detection of the
crease/patella at every frame. Validate visually and do not use for judging.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from math import hypot, isfinite
from pathlib import Path
from typing import Optional

from .analysis import FrameMetrics


def thigh_basis(hip: tuple[float, float], knee: tuple[float, float]):
    dx, dy = knee[0]-hip[0], knee[1]-hip[1]
    size = hypot(dx, dy)
    if size < 1e-5:
        raise ValueError('Hip and knee landmarks are too close for calibration.')
    ex = (dx/size, dy/size)
    ey = (-ex[1], ex[0])
    return size, ex, ey


def _coeff(point, origin, size, ex, ey):
    dx,dy = point[0]-origin[0],point[1]-origin[1]
    return ((dx*ex[0]+dy*ex[1])/size, (dx*ey[0]+dy*ey[1])/size)


def _project(origin, a, b, size, ex, ey):
    return (origin[0]+size*(a*ex[0]+b*ey[0]),
            origin[1]+size*(a*ex[1]+b*ey[1]))


@dataclass(frozen=True)
class DepthCalibration:
    """Relative coordinates tied to hip->knee segment, fixed to one side."""
    side: str
    crease_along: float
    crease_across: float
    knee_top_along: float
    knee_top_across: float
    reference_frame: int
    reference_time_s: float
    source_width: int
    source_height: int
    method: str = 'manual_two_point_thigh_attached'
    version: int = 2
    reference_hip_crease_px: Optional[tuple[float, float]] = None
    reference_knee_top_px: Optional[tuple[float, float]] = None
    # Additional manually verified anchors: (frame, crease_x,crease_y,knee_x,knee_y)
    additional_references: tuple[tuple[int,float,float,float,float], ...] = ()

    @classmethod
    def from_reference(cls, frame: FrameMetrics,
                       hip_crease_xy: tuple[float, float],
                       knee_top_xy: tuple[float, float],
                       width: int, height: int) -> 'DepthCalibration':
        """Input selected points in source-video pixel coordinates."""
        if width <= 0 or height <= 0 or not frame.points or frame.knee_deg is None:
            raise ValueError('A valid visible hip and knee pose is required for calibration.')
        if frame.side not in ('left', 'right'):
            raise ValueError('Calibration requires a fixed left or right side.')
        hip = (frame.points[1][0]*width,frame.points[1][1]*height)
        knee = (frame.points[2][0]*width,frame.points[2][1]*height)
        size, ex, ey = thigh_basis(hip,knee)
        for point, joint, title in [(hip_crease_xy,hip,'hip crease'),(knee_top_xy,knee,'knee top')]:
            if not all(isfinite(float(v)) for v in point):
                raise ValueError(f'{title} point contains invalid coordinates.')
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ValueError(f'{title} point must lie within video bounds.')
            if hypot(point[0]-joint[0],point[1]-joint[1]) > size*0.85:
                raise ValueError(f'{title} selection is too far from estimated joint; review landmark side.')
        ca,cv = _coeff(hip_crease_xy, hip, size, ex, ey)
        ka,kv = _coeff(knee_top_xy, knee, size, ex, ey)
        return cls(frame.side,ca,cv,ka,kv,frame.frame,frame.time_s,width,height,
                   method='manual_optical_flow_reference', version=2,
                   reference_hip_crease_px=tuple(hip_crease_xy),
                   reference_knee_top_px=tuple(knee_top_xy))

    def validate(self):
        if (self.version, self.method) not in ((1, 'manual_two_point_thigh_attached'),
                                                   (2, 'manual_optical_flow_reference')):
            raise ValueError('Unsupported depth-calibration format.')
        if self.side not in ('left', 'right'):
            raise ValueError('Calibration side must be left or right.')
        if self.source_width < 1 or self.source_height < 1:
            raise ValueError('Invalid calibration dimensions.')
        if self.version == 2:
            for name, p in [('hip crease', self.reference_hip_crease_px),
                            ('knee top', self.reference_knee_top_px)]:
                if p is None or len(p) != 2 or not all(isfinite(v) for v in p):
                    raise ValueError(f'Invalid {name} reference coordinates')
                if not (0 <= p[0] < self.source_width and 0 <= p[1] < self.source_height):
                    raise ValueError(f'{name} reference outside video bounds')
        seen={self.reference_frame}
        for item in self.additional_references:
            if len(item)!=5 or int(item[0])!=item[0] or item[0]<0 or item[0] in seen:
                raise ValueError('Additional reference frame duplicated or malformed')
            seen.add(item[0])
            _,hx,hy,kx,ky=item
            if not all(isfinite(v) for v in (hx,hy,kx,ky)) or not (
                0<=hx<self.source_width and 0<=kx<self.source_width and
                0<=hy<self.source_height and 0<=ky<self.source_height):
                raise ValueError('Additional reference coordinates outside frame')
        for value in (self.crease_along,self.crease_across,
                      self.knee_top_along,self.knee_top_across):
            if not isfinite(value) or abs(value) > 0.85:
                raise ValueError('Invalid/outlying depth calibration offsets.')

    def save(self, path: Path):
        self.validate()
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(self.__dict__,indent=2),encoding='utf-8')

    @classmethod
    def load(cls, path: Path):
        # Strictly validate keys to avoid silently accepting incompatible formats.
        payload = json.loads(path.read_text(encoding='utf-8'))
        required = {name for name, f in cls.__dataclass_fields__.items()
                    if f.default is __import__('dataclasses').MISSING}
        if not required.issubset(payload) or not set(payload).issubset(cls.__dataclass_fields__):
            raise ValueError('Calibration JSON has missing or unexpected fields.')
        if 'additional_references' in payload:
            payload['additional_references']=tuple(tuple(row) for row in payload['additional_references'])
        for key in ('reference_hip_crease_px','reference_knee_top_px'):
            if payload.get(key) is not None:
                payload[key] = tuple(payload[key])
        calibration = cls(**payload)
        calibration.validate()
        return calibration


def apply_depth_calibration(frame: FrameMetrics, calibration: DepthCalibration,
                            width: int, height: int) -> FrameMetrics:
    """Add per-frame anatomical reference estimates to a pose result.

    Returns an unavailable measurement when the chosen side is not visible.
    Does not interpolate across pose loss or infer depth from stale points.
    """
    if not frame.points or frame.knee_deg is None or frame.side != calibration.side:
        return replace(frame, crease_point=None, knee_top_point=None,
                       crease_to_knee_top_delta=None)
    if width != calibration.source_width or height != calibration.source_height:
        raise ValueError('Video dimensions differ from depth calibration. Recalibrate.')
    hip = (frame.points[1][0]*width,frame.points[1][1]*height)
    knee = (frame.points[2][0]*width,frame.points[2][1]*height)
    try:
        size,ex,ey = thigh_basis(hip,knee)
    except ValueError:
        return replace(frame, crease_point=None,knee_top_point=None,
                       crease_to_knee_top_delta=None)
    crease = _project(hip,calibration.crease_along,calibration.crease_across,size,ex,ey)
    top = _project(knee,calibration.knee_top_along,calibration.knee_top_across,size,ex,ey)
    crease_norm=(crease[0]/width,crease[1]/height)
    top_norm=(top[0]/width,top[1]/height)
    if not all(0 <= x <= 1 for x in (*crease_norm,*top_norm)):
        return replace(frame, crease_point=None,knee_top_point=None,
                       crease_to_knee_top_delta=None)
    return replace(frame,crease_point=crease_norm,knee_top_point=top_norm,
                   crease_to_knee_top_delta=crease_norm[1]-top_norm[1],
                   hip_crease_tracking_confidence=0., knee_top_tracking_confidence=0.,
                   depth_tracking_method='geometry_only_unverified')


def depth_status(delta: Optional[float], tolerance: float = 0.01) -> str:
    """Relative image-plane height; positive delta means crease estimate lower."""
    if delta is None:
        return 'unavailable'
    if delta > tolerance:
        return 'crease_below_knee_top'
    if delta < -tolerance:
        return 'crease_above_knee_top'
    return 'near_level_uncertain'


def classify_depth(delta_px: Optional[float], thigh_length_px: Optional[float],
                   confidence: Optional[float], uncertainty_px: float = 6.0) -> str:
    """Informational estimate, not IPF adjudication. Never classify weak tracking."""
    if delta_px is None or thigh_length_px is None or confidence is None:
        return 'unavailable'
    if confidence < .5 or thigh_length_px <= 0:
        return 'uncertain'
    margin = max(uncertainty_px, .025 * thigh_length_px)
    if abs(delta_px) <= margin:
        return 'near_threshold_uncertain'
    return 'below_estimate' if delta_px > 0 else 'above_estimate'
