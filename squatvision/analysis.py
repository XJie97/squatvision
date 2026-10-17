"""Offline squat movement segmentation with multi-signal, sustained-upright gating.

Design: only complete *validated* upright -> squat -> upright cycles count as reps.
Partial movements may become attempts, and an inferred failure is only a *suspected*
stall/reversal. No code here assesses competition depth or injury risk.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from statistics import median
from typing import Optional


@dataclass(frozen=True)
class FrameMetrics:
    frame: int
    time_s: float
    knee_deg: Optional[float] = None
    hip_deg: Optional[float] = None
    hip_y: Optional[float] = None
    knee_y: Optional[float] = None
    visibility: Optional[float] = None
    side: Optional[str] = None
    points: Optional[tuple[tuple[float, float], ...]] = None

    @property
    def depth_delta(self) -> Optional[float]:
        return (self.hip_y-self.knee_y) if self.hip_y is not None and self.knee_y is not None else None


@dataclass(frozen=True)
class Config:
    smoothing_radius: int = 4  # 9-frame median at 30fps (isolated noise suppression)
    min_knee_excursion_deg: float = 40.0
    min_hip_drop: float = 0.075  # Fraction of full image height; fixed camera required
    min_attempt_s: float = 0.85
    upright_hold_s: float = 0.30
    return_hold_s: float = 0.35
    upright_tolerance_deg: float = 12.0
    upright_hip_tolerance: float = 0.045
    movement_start_deg: float = 17.0
    movement_start_hip_drop: float = 0.023
    max_pose_gap_s: float = 0.50
    min_pause_s: float = 0.30
    min_ascent_hip_change: float = 0.020
    min_ascent_knee_change: float = 8.0
    failure_stall_s: float = 2.0


@dataclass(frozen=True)
class PhaseEvent:
    name: str
    time_s: float
    frame: int
    # Screenshot path added on export, not in the engine.

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Attempt:
    number: int
    outcome: str  # complete / incomplete / suspected_failure
    start_s: float
    end_s: float
    duration_s: float
    knee_min_deg: float
    hip_min_deg: Optional[float]
    hip_drop: float
    hip_below_knee_proxy: Optional[bool]
    bottom_pause_s: float
    events: list[PhaseEvent] = field(default_factory=list)
    note: str = ''

    def as_dict(self) -> dict:
        result = asdict(self)
        result['events'] = [e.as_dict() for e in self.events]
        return result


@dataclass
class Analysis:
    attempts: list[Attempt]
    smoothed_knee: list[Optional[float]]
    smoothed_hip_y: list[Optional[float]]
    upright_knee_deg: Optional[float]
    upright_hip_y: Optional[float]
    candidate_cycles_rejected: int = 0
    warning: str = ''

    @property
    def completed(self) -> list[Attempt]:
        return [a for a in self.attempts if a.outcome == 'complete']


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        raise ValueError('No values')
    pos = (len(xs)-1)*p
    i = int(pos)
    return xs[i] if i == len(xs)-1 else xs[i]*(1-(pos-i)) + xs[i+1]*(pos-i)


def median_smooth(values: list[Optional[float]], radius: int) -> list[Optional[float]]:
    """Smooth only locally contiguous valid samples; never interpolate missing points."""
    result = []
    for i, value in enumerate(values):
        if value is None:
            result.append(None)
            continue
        lo, hi = i, i
        while lo > max(0, i-radius) and values[lo-1] is not None:
            lo -= 1
        while hi < min(len(values)-1, i+radius) and values[hi+1] is not None:
            hi += 1
        neighbours = [v for v in values[lo:hi+1] if v is not None]
        result.append(float(median(neighbours)))
    return result


def _event(name: str, frames: list[FrameMetrics], i: int) -> PhaseEvent:
    return PhaseEvent(name, round(frames[i].time_s, 3), frames[i].frame)


def _phases(frames: list[FrameMetrics], knee: list[Optional[float]], hip: list[Optional[float]],
            a: int, b: int, ref_knee: float, base_hip: float, config: Config,
            complete: bool) -> tuple[list[PhaseEvent], float, bool]:
    """Derive the bottom and ascent using smoothed movement trajectory."""
    ids = [i for i in range(a,b+1) if knee[i] is not None and hip[i] is not None]
    if not ids:
        return [_event('descent_start', frames, a)], 0., False
    min_knee = min(knee[i] for i in ids)
    max_hip = max(hip[i] for i in ids)
    # Bottom: deepest measured hip position, with strong knee flexion. Earliest
    # qualifying point avoids placing descent_end at the end of a long hold.
    candidates = [i for i in ids if hip[i] >= max_hip - 0.012 and
                  knee[i] <= min_knee + 12.]
    if not candidates:
        candidates = [max(ids, key=lambda i: (hip[i]-base_hip)*2 + (ref_knee-knee[i])/150)]
    bottom = candidates[0]
    events = [_event('descent_start', frames, a), _event('descent_end', frames, bottom)]
    # Require BOTH visibly rising hips and knee extension, sustained over time.
    ascent = None
    for i in ids:
        if i <= bottom or knee[i] < min_knee + 3.0 or hip[i] > max_hip - 0.007:
            continue
        future = [j for j in ids if i <= j <= i+10]
        if future and max(knee[j] for j in future) >= min_knee+config.min_ascent_knee_change and \
                min(hip[j] for j in future) <= max_hip-config.min_ascent_hip_change:
            ascent = i
            break
    pause = 0.0
    if ascent is not None:
        events.append(_event('ascent_start', frames, ascent))
        pause = max(0., frames[ascent].time_s-frames[bottom].time_s)
        if pause < config.min_pause_s:
            pause = 0.0
    if complete:
        events.append(_event('lockout', frames, b))
    return events, round(pause, 2), ascent is not None


def _failure_frame(frames: list[FrameMetrics], knee: list[Optional[float]],
                   hip: list[Optional[float]], events: list[PhaseEvent],
                   b: int, ref_knee: float, config: Config) -> Optional[int]:
    """Only label *suspected* failure when an attempted ascent reversed or stalled.

    Cropped clips, absent pose data and long bottom holds are not evidence of failure.
    """
    ascent = next((e.frame for e in events if e.name == 'ascent_start'), None)
    if ascent is None:
        return None
    frame_index = {f.frame: i for i,f in enumerate(frames)}
    a = frame_index[ascent]
    valid = [i for i in range(a,b+1) if knee[i] is not None and hip[i] is not None]
    if len(valid) < 5:
        return None
    best_i = max(valid, key=lambda i: knee[i])
    # Reversed ascent: gain > 15deg, then flex > 14deg again, remaining short of upright.
    end_i = valid[-1]
    if knee[best_i] >= knee[a]+15 and best_i < end_i and \
        knee[end_i] <= knee[best_i]-14 and hip[end_i] >= hip[best_i]+0.028 and \
        knee[end_i] < ref_knee-config.upright_tolerance_deg:
        return best_i
    # Stall while still flexed after ascent began. Conservative: at least 2s
    # with almost no progress at the *end*, and no recovered upright position.
    tail = [i for i in valid if frames[end_i].time_s-frames[i].time_s <= config.failure_stall_s]
    if len(tail) >= 5 and frames[tail[-1]].time_s-frames[tail[0]].time_s >= config.failure_stall_s*0.95:
        if max(knee[i] for i in tail)-min(knee[i] for i in tail) < 7 and \
            max(hip[i] for i in tail)-min(hip[i] for i in tail) < 0.023 and \
            knee[end_i] < ref_knee-config.upright_tolerance_deg:
            return tail[0]
    return None


def analyse(frames: list[FrameMetrics], config: Config | None = None) -> Analysis:
    config = config or Config()
    if not frames:
        return Analysis([], [], [], None, None, warning='No frames provided.')
    if any(frames[i].time_s >= frames[i+1].time_s for i in range(len(frames)-1)):
        raise ValueError('Video timestamps must be strictly increasing')
    knee = median_smooth([f.knee_deg if f.hip_y is not None else None for f in frames],
                         config.smoothing_radius)
    hip = median_smooth([f.hip_y if f.knee_deg is not None else None for f in frames],
                        config.smoothing_radius)
    valid = [i for i in range(len(frames)) if knee[i] is not None and hip[i] is not None]
    if len(valid) < 10:
        return Analysis([], knee, hip, None, None, warning='Too few usable pose frames.')
    ref_knee = percentile([knee[i] for i in valid], 0.90)
    high_knee = [i for i in valid if knee[i] >= ref_knee-9]
    base_hip = median([hip[i] for i in high_knee])
    warning = ''
    if ref_knee < 135:
        warning = ('Upright reference is unusually flexed; check camera angle, '
                   'visibility and side selection.')

    def upright(i: int) -> bool:
        return (knee[i] is not None and hip[i] is not None and
                knee[i] >= ref_knee-config.upright_tolerance_deg and
                hip[i] <= base_hip+config.upright_hip_tolerance)

    attempts: list[Attempt] = []
    rejected = 0
    mode = 'seek_standing'
    standing_first = None
    last_standing = None
    start = None
    return_first = None
    missing_first = None

    def close_attempt(end: int, complete: bool):
        nonlocal rejected, start
        if start is None or end <= start:
            return
        idx = [i for i in range(start,end+1) if knee[i] is not None and hip[i] is not None]
        if not idx:
            rejected += 1
            return
        knee_min = min(knee[i] for i in idx)
        max_drop = max(hip[i] for i in idx) - hip[start]
        duration = frames[end].time_s-frames[start].time_s
        if ref_knee-knee_min < config.min_knee_excursion_deg or \
                max_drop < config.min_hip_drop or duration < config.min_attempt_s:
            rejected += 1
            return
        events, pause, ascent_started = _phases(
            frames,knee,hip,start,end,ref_knee,base_hip,config,complete)
        outcome = 'complete' if complete else 'incomplete'
        failure_index = None
        if not complete and ascent_started:
            failure_index = _failure_frame(frames,knee,hip,events,end,ref_knee,config)
            if failure_index is not None:
                outcome = 'suspected_failure'
                events.append(_event('suspected_failure', frames, failure_index))
        for e in events:
            if e.name == 'lockout' and not complete:
                raise AssertionError('Incomplete attempts cannot have lockout events')
        delta = [frames[i].depth_delta for i in idx if frames[i].depth_delta is not None]
        hip_angles = [frames[i].hip_deg for i in idx if frames[i].hip_deg is not None]
        # hip joint landmark vs knee joint landmark - a perspective-dependent proxy.
        depth = max(delta) >= 0.015 if delta else None
        attempts.append(Attempt(
            number=len(attempts)+1, outcome=outcome,
            start_s=round(frames[start].time_s,3), end_s=round(frames[end].time_s,3),
            duration_s=round(duration,2), knee_min_deg=round(knee_min,1),
            hip_min_deg=round(min(hip_angles),1) if hip_angles else None,
            hip_drop=round(max_drop,4), hip_below_knee_proxy=depth,
            bottom_pause_s=pause, events=events,
            note=('Estimated standing return, not verified joint lockout' if complete else
                  'Heuristic detection, manual review necessary' if failure_index is not None else
                  'No sustained standing return captured; may be a cropped clip'),
        ))

    for i, f in enumerate(frames):
        usable = knee[i] is not None and hip[i] is not None
        if not usable:
            if missing_first is None:
                missing_first = i
            elif frames[i].time_s-frames[missing_first].time_s > config.max_pose_gap_s:
                if mode == 'moving' and start is not None:
                    close_attempt(missing_first, complete=False)
                mode, standing_first, last_standing, start, return_first = \
                    'seek_standing', None, None, None, None
            continue
        if missing_first is not None:
            if frames[i].time_s - frames[missing_first].time_s > config.max_pose_gap_s:
                if mode == 'moving' and start is not None:
                    close_attempt(missing_first, complete=False)
                mode, standing_first, last_standing, start, return_first = \
                    'seek_standing', None, None, None, None
            missing_first = None
        if mode in ('seek_standing','ready'):
            if upright(i):
                if standing_first is None:
                    standing_first = i
                if frames[i].time_s-frames[standing_first].time_s >= config.upright_hold_s:
                    mode = 'ready'
                    last_standing = i
            else:
                standing_first = None
                if mode == 'ready' and last_standing is not None and \
                   knee[i] <= ref_knee-config.movement_start_deg and \
                   hip[i] >= base_hip+config.movement_start_hip_drop:
                    # Begin at most recent fully upright sample before descent.
                    start = last_standing
                    return_first = None
                    mode = 'moving'
        elif mode == 'moving':
            if upright(i):
                if return_first is None:
                    return_first = i
                if frames[i].time_s-frames[return_first].time_s >= config.return_hold_s:
                    close_attempt(return_first, complete=True)
                    mode = 'ready'
                    standing_first = return_first
                    last_standing = i
                    return_first = None
                    start = None
            else:
                return_first = None
    if mode == 'moving' and start is not None:
        close_attempt(len(frames)-1, complete=False)
    return Analysis(attempts, knee, hip, round(ref_knee,2), round(base_hip,4), rejected, warning)
