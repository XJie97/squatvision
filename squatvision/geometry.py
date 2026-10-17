"""2D pose geometry. Results are view-dependent image measurements, not 3D biomechanics."""
from __future__ import annotations
from math import acos, degrees, hypot
from typing import Sequence
from .analysis import FrameMetrics

# MediaPipe Pose Landmarker indices: shoulder, hip, knee, ankle
JOINTS = {'left': (11, 23, 25, 27), 'right': (12, 24, 26, 28)}


def angle_degrees(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float | None:
    u = (a[0]-b[0], a[1]-b[1])
    v = (c[0]-b[0], c[1]-b[1])
    nu, nv = hypot(*u), hypot(*v)
    if min(nu, nv) < 1e-8:
        return None
    return degrees(acos(max(-1., min(1., (u[0]*v[0]+u[1]*v[1])/(nu*nv)))))


def best_side(landmarks: Sequence, min_visibility: float = 0.55) -> str | None:
    candidates = []
    for name, ids in JOINTS.items():
        visibility = [float(getattr(landmarks[i], 'visibility', 1.) or 0.) for i in ids]
        if min(visibility) >= min_visibility:
            candidates.append((sum(visibility), name))
    return max(candidates)[1] if candidates else None


def extract_frame(landmarks: Sequence, index: int, t: float,
                  width: int, height: int, side: str,
                  min_visibility: float = 0.55) -> FrameMetrics:
    lm = [landmarks[i] for i in JOINTS[side]]
    visibility = min(float(getattr(l, 'visibility', 1.) or 0.) for l in lm)
    if visibility < min_visibility:
        return FrameMetrics(frame=index, time_s=t, visibility=visibility, side=side)
    pts = [(float(p.x), float(p.y)) for p in lm]
    px = [(x*width, y*height) for x,y in pts]  # Fix x/y aspect-ratio distortion.
    shoulder, hip, knee, ankle = px
    return FrameMetrics(
        frame=index, time_s=t, knee_deg=angle_degrees(hip, knee, ankle),
        hip_deg=angle_degrees(shoulder, hip, knee),
        hip_y=pts[1][1], knee_y=pts[2][1],
        visibility=visibility, side=side, points=tuple(pts),
    )
