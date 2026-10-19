"""Independent 2-D anatomical-reference tracking from manually clicked image points.

Uses local Shi-Tomasi features and forward/backward-checked pyramidal LK flow.
No point is reconstructed from joint-centre Y alone. An untrackable point is
unavailable, not silently replaced by a plausible-looking extrapolation.
"""
from __future__ import annotations

from dataclasses import replace
from math import hypot
from pathlib import Path
from typing import Optional

import cv2
from squatvision.video import open_video
import numpy as np

from .analysis import FrameMetrics
from .depth import DepthCalibration


class LocalPointTracker:
    """One local feature constellation, relative to one independent manual point."""

    def __init__(self, gray: np.ndarray, xy: tuple[float, float], radius: int = 27):
        self.xy = np.asarray(xy, dtype=np.float32)
        self.prev = gray
        self.radius = radius
        self.features = self._features(gray, self.xy)
        self.active = len(self.features) >= 4
        self.confidence = 1.0 if self.active else 0.0
        self.lost_reason = '' if self.active else 'insufficient_local_texture'
        self.steps = 0

    def _features(self, gray: np.ndarray, xy: np.ndarray):
        h,w=gray.shape[:2]
        mask=np.zeros((h,w),np.uint8)
        cv2.circle(mask,tuple(np.round(xy).astype(int)),self.radius,255,-1)
        pts=cv2.goodFeaturesToTrack(gray,maxCorners=45,qualityLevel=.018,
                                    minDistance=3,blockSize=5,mask=mask)
        return np.empty((0,2),np.float32) if pts is None else pts.reshape(-1,2).astype(np.float32)

    def update(self, gray: np.ndarray, pose_expected: Optional[tuple[float,float]]=None,
               max_pose_disagreement: float = 90.) -> None:
        """Estimate affine local motion; never reinitialise an already-lost anchor."""
        if not self.active:
            self.prev=gray
            return
        p0=self.features.reshape(-1,1,2)
        p1,st,_=cv2.calcOpticalFlowPyrLK(self.prev,gray,p0,None,
                                         winSize=(23,23),maxLevel=3,
                                         criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
        if p1 is None:
            self._lose('optical_flow_failed',gray)
            return
        back,back_ok,_=cv2.calcOpticalFlowPyrLK(gray,self.prev,p1,None,
                                               winSize=(23,23),maxLevel=3)
        if back is None:
            self._lose('optical_flow_backward_failed',gray)
            return
        errors=np.linalg.norm(back.reshape(-1,2)-p0.reshape(-1,2),axis=1)
        good=(st.ravel()!=0)&(back_ok.ravel()!=0)&(errors<1.6)
        x=p0.reshape(-1,2)[good]
        y=p1.reshape(-1,2)[good]
        if len(x)<4:
            self._lose('too_few_consistent_features',gray)
            return
        affine,inlier=cv2.estimateAffinePartial2D(x,y,method=cv2.RANSAC,
                                                  ransacReprojThreshold=2.2,
                                                  maxIters=200,confidence=.99)
        if affine is None or inlier is None or int(inlier.sum())<4:
            self._lose('local_motion_inconsistent',gray)
            return
        inliers=inlier.ravel().astype(bool)
        # 2-D estimate of the anatomical anchor using coherent local image motion.
        candidate=(affine[:,:2]@self.xy)+affine[:,2]
        if not np.isfinite(candidate).all() or np.linalg.norm(candidate-self.xy)>38:
            self._lose('implausible_frame_displacement',gray)
            return
        h,w=gray.shape[:2]
        if not (0<=candidate[0]<w and 0<=candidate[1]<h):
            self._lose('point_left_frame',gray)
            return
        if pose_expected is not None and hypot(*(candidate-np.asarray(pose_expected)))>max_pose_disagreement:
            self._lose('inconsistent_with_body_pose',gray)
            return
        kept=y[inliers]
        if len(kept)<6:
            self._lose('insufficient_inlier_features',gray)
            return
        # Confidence must reflect *track quality*, not an arbitrary target
        # count of 14 features. A low-texture shorts/hip ROI can contain just
        # six coherent corners. Previously 6/14 = 0.429 passed the tracker's
        # 0.4 threshold but failed track_anatomy's 0.5 export threshold,
        # making the hip marker disappear immediately after a good manual
        # annotation (confirmed in the real squat at 72.724 -> 72.758 s).
        # Require >=6 robust inliers, high retention, and low forward/backward
        # error. This is motion quality, NOT anatomical certainty.
        n=len(kept)
        retention=n/max(1,len(p0))
        median_fb=float(np.median(errors[good][inliers]))
        # Six features are enough to estimate stable local motion but offer
        # less redundancy than a highly textured ROI. 0.65 is a *heuristic*
        # initial support score, not a probability of anatomical accuracy.
        support=min(1.0,0.65+0.035*max(0,n-6))
        fb_quality=max(0.0,1.0-median_fb/3.0)
        quality=float(support*np.sqrt(retention)*fb_quality)
        if quality < .5:
            self._lose('tracking_confidence_low',gray)
            return
        self.xy=candidate.astype(np.float32)
        self.features=kept.astype(np.float32)
        self.confidence=quality
        self.prev=gray
        self.steps+=1

    def _lose(self, reason, gray):
        self.active=False
        self.confidence=0.0
        self.lost_reason=reason
        self.prev=gray


def _reference_points(cal: DepthCalibration):
    if cal.version != 2 or cal.reference_hip_crease_px is None or cal.reference_knee_top_px is None:
        raise ValueError('Legacy calibration lacks exact image points. Re-run calibrate_depth.py.')
    return cal.reference_hip_crease_px, cal.reference_knee_top_px


def _pose_xy(frame: FrameMetrics, joint_id: int, w: int, h: int):
    return (frame.points[joint_id][0]*w,frame.points[joint_id][1]*h) if frame.points else None


def _read_gray(cap: cv2.VideoCapture, frame_id: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
    ok,img=cap.read()
    if not ok:
        raise RuntimeError(f'Could not decode source frame {frame_id}')
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


def track_anatomy(video: Path, frames: list[FrameMetrics], cal: DepthCalibration,
                  width: int, height: int) -> list[FrameMetrics]:
    """Track independently from calibration time in both time directions.

    Frame seeking (for backwards traversal) trades speed for bounded RAM. A
    missing pose never creates an inferred depth value, even if flow continues.
    """
    if not frames:
        return []
    if cal.side not in {'left','right'} or cal.source_width!=width or cal.source_height!=height:
        raise ValueError('Calibration does not match video dimensions/side.')
    hip_xy,knee_xy=_reference_points(cal)
    ordered={f.frame:f for f in frames}
    ref=cal.reference_frame
    anchor_ids=[cal.reference_frame,*[row[0] for row in cal.additional_references]]
    if not any(i in ordered for i in anchor_ids):
        raise ValueError('At least one manual reference must be inside the analysed window.')
    for i in anchor_ids:
        if i in ordered and (ordered[i].side!=cal.side or ordered[i].points is None):
            raise ValueError('No valid corresponding body pose on reference frame.')
    cap=open_video(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open source video {video}')
    try:
        anchors=[(cal.reference_frame,*hip_xy,*knee_xy),*cal.additional_references]
        anchors=sorted(anchors,key=lambda item:item[0])
        valid_anchors=[a for a in anchors if a[0] in ordered]
        if not valid_anchors:
            raise ValueError('No manual calibration anchor lies within the analysed time window.')
        # Every analysed frame belongs to its *nearest* manually verified anchor.
        # Thus lost tracks can be reinitialised only from an independently clicked
        # frame, never from a fabricated predicted anatomical position.
        frame_ids=sorted(ordered)
        assignments={a[0]:[] for a in valid_anchors}
        for i in frame_ids:
            nearest=min(valid_anchors,key=lambda a:(abs(i-a[0]),a[0]))
            assignments[nearest[0]].append(i)
        # Cache short segments by decoding sequentially; repeatedly seeking to
        # individual MP4 frames is extremely slow (especially backwards).
        # Limit RAM for long videos; recommended CLI analysis window is 64..80s.
        lo,hi=min(frame_ids),max(frame_ids)
        gray_cache={}
        if hi-lo<=600:
            cap.set(cv2.CAP_PROP_POS_FRAMES,lo)
            for i in range(lo,hi+1):
                ok,img=cap.read()
                if not ok:
                    raise RuntimeError(f'Could not decode source frame {i}')
                gray_cache[i]=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        def gray_at(i):
            return gray_cache[i] if i in gray_cache else _read_gray(cap,i)
        result={}
        for anchor in valid_anchors:
            ref,hipx,hipy,kneex,kneey=anchor
            reference=gray_at(ref)
            reference_pose=ordered[ref]
            # Manual points are separate from pose joint centres. Pose motion is
            # only a loose sanity gate, never used as replacement XY output.
            reference_hip=_pose_xy(reference_pose,1,width,height)
            reference_knee=_pose_xy(reference_pose,2,width,height)
            thigh_len=hypot(reference_hip[0]-reference_knee[0],
                            reference_hip[1]-reference_knee[1])
            pose_gate_px=max(12.,.16*thigh_len)
            ids=assignments[ref]
            for direction,subids in ((1,[i for i in ids if i>=ref]),
                                      (-1,[i for i in reversed(ids) if i<ref])):
                hip=LocalPointTracker(reference,(hipx,hipy))
                knee=LocalPointTracker(reference,(kneex,kneey))
                prev_id=ref
                for i in subids:
                    f=ordered[i]
                    if i!=ref:
                        for intermediate in range(prev_id+direction,i+direction,direction):
                            gray=gray_at(intermediate)
                            pose=ordered.get(intermediate)
                            ph=_pose_xy(pose,1,width,height) if pose and pose.side==cal.side else None
                            pk=_pose_xy(pose,2,width,height) if pose and pose.side==cal.side else None
                            if ph is not None:
                                ph=(hipx+ph[0]-reference_hip[0],
                                    hipy+ph[1]-reference_hip[1])
                            if pk is not None:
                                pk=(kneex+pk[0]-reference_knee[0],
                                    kneey+pk[1]-reference_knee[1])
                            hip.update(gray,ph,max_pose_disagreement=pose_gate_px)
                            knee.update(gray,pk,max_pose_disagreement=pose_gate_px)
                        prev_id=i
                    pose_ok=f.side==cal.side and f.points is not None and f.knee_deg is not None
                    hc=tuple((hip.xy/np.array([width,height])).tolist()) if pose_ok and hip.active and hip.confidence>=.5 else None
                    kt=tuple((knee.xy/np.array([width,height])).tolist()) if pose_ok and knee.active and knee.confidence>=.5 else None
                    depth=(hc[1]-kt[1]) if hc is not None and kt is not None else None
                    reason=('' if depth is not None else
                            'missing_or_wrong_side_pose' if not pose_ok else
                            'hip_crease_'+hip.lost_reason if not hip.active else
                            'knee_top_'+knee.lost_reason if not knee.active else 'low_visual_tracking_confidence')
                    result[i]=replace(f,crease_point=hc,knee_top_point=kt,
                                      crease_to_knee_top_delta=depth,
                                      hip_crease_tracking_confidence=hip.confidence if pose_ok else 0.,
                                      knee_top_tracking_confidence=knee.confidence if pose_ok else 0.,
                                      depth_tracking_method='local_optical_flow',
                                      depth_unavailable_reason=reason)
        return [result[f.frame] for f in frames]
    finally:
        cap.release()


def calibration_from_manual_annotations(
    frames: list[FrameMetrics],
    manual_points: dict[int, object],
    width: int,
    height: int,
) -> tuple[DepthCalibration | None, str]:
    """Build a transient flow calibration from reviewed manual surface labels.

    This runs ONLY for anatomically labelled frames where the matching selected
    MediaPipe body side and joints are available. The calibration is not written
    to disk; the original manual JSON remains the canonical annotations.

    No calibration is fabricated when manual anchors fail geometric validation.
    """
    lookup={f.frame:f for f in frames}
    accepted=[]
    rejected=[]
    for i, label in sorted(manual_points.items()):
        frame=lookup.get(i)
        if frame is None:
            continue
        if not frame.points or frame.side not in ('left','right') or frame.knee_deg is None:
            rejected.append(f'{i}: no matching pose')
            continue
        try:
            cal=DepthCalibration.from_reference(
                frame, label.upper_thigh_at_hip, label.top_of_knee, width, height)
        except ValueError as exc:
            rejected.append(f'{i}: {exc}')
            continue
        accepted.append((i,label,cal))
    if not accepted:
        return None, ('No usable manual anchors within the analysed video window. '
                      'Manually annotated frames remain visible; other frames are '
                      'unavailable. ' + '; '.join(rejected[:3]))
    first=accepted[0][2]
    candidates=[(i,lab) for i,lab,_ in accepted if lab is not accepted[0][1]]
    anchors=tuple((i,*lab.upper_thigh_at_hip,*lab.top_of_knee)
                  for i,lab in candidates if lookup[i].side==first.side)
    cal=replace(first,additional_references=anchors)
    cal.validate()
    warning=(f'{len(accepted)} manual anchors available; visual flow is provisional. '
             f'{len(rejected)} rejected anchor(s).' if rejected else '')
    return cal, warning
