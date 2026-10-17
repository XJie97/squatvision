"""Video and event-screenshot overlays; dashed angular guides preserve joint visibility."""
from __future__ import annotations
from dataclasses import dataclass
from math import atan2, degrees, hypot
from pathlib import Path
import cv2
import numpy as np
from .analysis import Analysis, FrameMetrics, PhaseEvent
from .depth import depth_status
from .ipf_depth import measure, UNAVAILABLE, BELOW, ABOVE, AUTOMATED

KNEE_COLOUR = (255, 229, 70)  # Cyan-yellow in OpenCV BGR
HIP_COLOUR = (68, 164, 255)   # Orange
TEXT = (245, 245, 245)


def _dash_arm(canvas: np.ndarray, center: tuple[int,int], point: tuple[int,int],
              color: tuple[int,int,int], start_px: float = 19.,
              length_px: float = 76., dash_px: float = 6., gap_px: float = 6., offset: float = 0.):
    dx,dy = point[0]-center[0], point[1]-center[1]
    dist = hypot(dx,dy)
    if dist < start_px+8:
        return
    ux,uy=dx/dist,dy/dist
    nx,ny=-uy,ux
    end=min(dist-9.,length_px)
    pos=start_px
    while pos<end:
        next_pos=min(pos+dash_px,end)
        a=(int(round(center[0]+pos*ux+offset*nx)),int(round(center[1]+pos*uy+offset*ny)))
        b=(int(round(center[0]+next_pos*ux+offset*nx)),int(round(center[1]+next_pos*uy+offset*ny)))
        cv2.line(canvas,a,b,color,2,cv2.LINE_AA)
        pos+=dash_px+gap_px


def _angle_arc(canvas: np.ndarray, point_a, joint, point_b, color):
    # A short angle arc complements the dashed rays, but never covers the joint.
    v1=(point_a[0]-joint[0],point_a[1]-joint[1])
    v2=(point_b[0]-joint[0],point_b[1]-joint[1])
    if min(hypot(*v1),hypot(*v2)) < 25:
        return
    p=degrees(atan2(v1[1],v1[0]))
    q=degrees(atan2(v2[1],v2[0]))
    sweep=(q-p+180)%360-180
    cv2.ellipse(canvas,joint,(27,27),0,p,p+sweep,color,1,cv2.LINE_AA)


def draw_angle_guides(image: np.ndarray, frame: FrameMetrics):
    """Draw small separated dashes along angle arms, avoiding centre 19px of joints."""
    if not frame.points or frame.knee_deg is None:
        return
    height,width=image.shape[:2]
    shoulder,hip,knee,ankle = [(int(round(x*width)),int(round(y*height))) for x,y in frame.points]
    # Hip angle between torso and femur. Offset from femur centreline so both
    # hip and knee measurement guides remain visible on the same limb segment.
    _dash_arm(image,hip,shoulder,HIP_COLOUR,offset=2)
    _dash_arm(image,hip,knee,HIP_COLOUR,offset=3)
    _angle_arc(image,shoulder,hip,knee,HIP_COLOUR)
    # Knee angle between thigh and lower leg.
    _dash_arm(image,knee,hip,KNEE_COLOUR,offset=-3)
    _dash_arm(image,knee,ankle,KNEE_COLOUR,offset=-2)
    _angle_arc(image,hip,knee,ankle,KNEE_COLOUR)
    for (joint,colour) in [(hip,HIP_COLOUR),(knee,KNEE_COLOUR)]:
        cv2.circle(image,joint,6,(30,32,35),1,cv2.LINE_AA)
        cv2.circle(image,joint,3,colour,1,cv2.LINE_AA)
    # Numeric angles live in the HUD to avoid covering the athlete and joints.


def timestamp(t: float) -> str:
    minutes = int(t//60)
    seconds = t-minutes*60
    return f'{minutes:02d}:{seconds:06.3f}'


def _fitted_text(image: np.ndarray, message: str, y: int,
                 max_width: int, colour=TEXT, initial_scale: float=.45) -> None:
    """Fit the full status within narrow source footage (never truncate context)."""
    scale=initial_scale
    while scale>.27:
        (tw,_),_=cv2.getTextSize(message,cv2.FONT_HERSHEY_SIMPLEX,scale,1)
        if tw<=max_width:break
        scale-=.02
    cv2.putText(image,message,(7,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(4,8,12),3,cv2.LINE_AA)
    cv2.putText(image,message,(7,y),cv2.FONT_HERSHEY_SIMPLEX,scale,colour,1,cv2.LINE_AA)


def overlay(frame: np.ndarray, point: FrameMetrics, analysis: Analysis,
            event_label: str | None = None) -> np.ndarray:
    result=frame.copy()
    draw_angle_guides(result,point)
    draw_depth_guides(result,point)
    h,w=result.shape[:2]
    completed=sum(a.outcome=='complete' and a.end_s<=point.time_s for a in analysis.attempts)
    m=measure(point,w,h,require_manual_verification=True)
    provenance={'MANUAL_ANNOTATION':'MANUAL','MANUAL_VERIFIED_TRACKING':'VERIFIED',
                'AUTOMATED_TRACKING':'FLOW'}.get(m.measurement_source,'NONE')
    if m.depth_margin_px is None:
        lines=[f'{timestamp(point.time_s)}  |  Reps {completed}',
               'Depth: UNAVAILABLE',
               'Surface references not reliable']
    elif m.measurement_source == AUTOMATED:
        lines=[f'{timestamp(point.time_s)}  |  Reps {completed}',
               f'Depth proxy: {m.depth_margin_px:+.1f}px (FLOW)',
               'UNVERIFIED - review bottom frame']
    else:
        short={BELOW:'EST. BELOW',ABOVE:'OBS. ABOVE'}.get(m.depth_status,'BORDERLINE')
        lines=[f'{timestamp(point.time_s)}  |  Reps {completed}',
               f'IPF surface margin {m.depth_margin_px:+.1f}px',
               f'{short}  +/-{m.depth_uncertainty_px:.1f}px [{provenance}]']
    angle_line=(f'Knee {point.knee_deg:.0f} / Hip {point.hip_deg:.0f} deg'
                if point.knee_deg is not None and point.hip_deg is not None else
                'Joint angles unavailable')
    lines.append(angle_line)
    if event_label:lines.append('EVENT: '+event_label.replace('_',' ').upper())
    # Compact, fixed-height panel; full strings are font-scaled, not clipped.
    panel_h=min(h,8+len(lines)*20)
    dark=result.copy()
    cv2.rectangle(dark,(0,0),(w,panel_h),(9,17,23),-1)
    cv2.addWeighted(dark,.85,result,.15,0,result)
    for i,line in enumerate(lines):
        tone=(85,220,255) if i==len(lines)-1 and event_label else TEXT
        _fitted_text(result,line,17+20*i,w-12,tone,.46 if w<=420 else .58)
    # Small unobtrusive legend below frame content.
    if h>160:
        cv2.putText(result,'Orange hip / Cyan knee: joint centres',
                    (8,h-8),cv2.FONT_HERSHEY_SIMPLEX,.36,(244,244,244),1,cv2.LINE_AA)
    return result


def screenshot_name(attempt: int, event: PhaseEvent) -> str:
    stamp=f'{event.time_s:010.3f}'.replace('.','p')
    return f'attempt_{attempt:02d}_{event.name}_{stamp}s.png'

# Depth overlay uses independent anatomical reference-point *estimates*.
# NEVER label MediaPipe hip/knee joint centres as the top hip crease/knee surface.
DEPTH_COLOUR = (95, 215, 105)  # Green for calibrated depth measurement
DEPTH_REFERENCE_COLOUR = (185, 180, 245)


def draw_depth_guides(image: np.ndarray, frame: FrameMetrics):
    """Reference positions with short dashed guides, no anatomy-obscuring labels.

    These pixels visualise anatomical surface estimates, not pose joint centres.
    """
    if frame.crease_point is None or frame.knee_top_point is None:
        return
    height,width=image.shape[:2]
    hip=(round(frame.crease_point[0]*width),round(frame.crease_point[1]*height))
    knee=(round(frame.knee_top_point[0]*width),round(frame.knee_top_point[1]*height))
    if not all(0<=p[0]<width and 0<=p[1]<height for p in (hip,knee)):
        return
    # Dashed knee-height marker local to the knee, not across the whole screen.
    y=knee[1]
    for begin,end in ((max(0,knee[0]-85),max(0,knee[0]-12)),
                      (min(width-1,knee[0]+12),min(width-1,knee[0]+45))):
        for x in range(begin,end,11):
            cv2.line(image,(x,y),(min(x+5,end),y),DEPTH_REFERENCE_COLOUR,1,cv2.LINE_AA)
    # Bracket positioned alongside the reference points, not on the hip/knee.
    bracket_x=min(width-10,max(hip[0],knee[0])+23)
    for (x,yy),col in ((hip,DEPTH_COLOUR),(knee,DEPTH_REFERENCE_COLOUR)):
        cv2.circle(image,(x,yy),6,(15,15,15),2,cv2.LINE_AA)
        cv2.circle(image,(x,yy),5,col,1,cv2.LINE_AA)
        direction=1 if bracket_x>x else -1
        xx=x+direction*9
        while (xx<bracket_x-5 if direction>0 else xx>bracket_x+5):
            endx=xx+direction*4
            cv2.line(image,(xx,yy),(endx,yy),col,1,cv2.LINE_AA)
            xx+=direction*10
    if abs(hip[1]-knee[1])>3:
        lo,hi=sorted([hip[1],knee[1]])
        for yy in range(lo,hi,10):
            cv2.line(image,(bracket_x,yy),(bracket_x,min(yy+5,hi)),DEPTH_COLOUR,1,cv2.LINE_AA)
    # Tiny point identifiers without misleading joint-angle labels.
    for title,point,col in [('H',hip,DEPTH_COLOUR),('K',knee,DEPTH_REFERENCE_COLOUR)]:
        xx=point[0]-18 if point[0]>35 else point[0]+9
        yy=max(100,min(height-8,point[1]-9))
        cv2.putText(image,title,(xx,yy),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(image,title,(xx,yy),cv2.FONT_HERSHEY_SIMPLEX,.4,col,1,cv2.LINE_AA)
