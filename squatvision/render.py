"""Video and event-screenshot overlays; dashed angular guides preserve joint visibility."""
from __future__ import annotations
from dataclasses import dataclass
from math import atan2, degrees, hypot
from pathlib import Path
import cv2
import numpy as np
from .analysis import Analysis, FrameMetrics, PhaseEvent
from .depth import depth_status

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
    # Labels positioned off the joints, inside image bounds. Draw a dark border.
    for joint,value,colour,xoff,yoff in [(hip,frame.hip_deg,HIP_COLOUR,12,-12),
                                        (knee,frame.knee_deg,KNEE_COLOUR,12,20)]:
        if value is None:
            continue
        text=f'{value:.0f} deg'
        x=max(3,min(width-100,joint[0]+xoff))
        y=max(20,min(height-20,joint[1]+yoff))
        cv2.putText(image,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(image,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.55,colour,2,cv2.LINE_AA)


def timestamp(t: float) -> str:
    minutes = int(t//60)
    seconds = t-minutes*60
    return f'{minutes:02d}:{seconds:06.3f}'


def overlay(frame: np.ndarray, point: FrameMetrics, analysis: Analysis,
            event_label: str | None = None) -> np.ndarray:
    result=frame.copy()
    draw_angle_guides(result,point)
    draw_depth_guides(result,point)
    h,w=result.shape[:2]
    padding=8
    panel_h=min(h,166)
    base=result.copy()
    cv2.rectangle(base,(0,0),(min(w,610),panel_h),(18,25,32),-1)
    cv2.addWeighted(base,0.68,result,0.32,0,result)
    completed=sum(a.outcome=='complete' and a.end_s<=point.time_s for a in analysis.attempts)
    rows=[f'Time {timestamp(point.time_s)} | Reps {completed}',
          f'Knee {point.knee_deg:.1f} deg | Hip {point.hip_deg:.1f} deg'
          if point.knee_deg is not None and point.hip_deg is not None else 'Pose unavailable / low confidence',
          f'Side {point.side or "unknown"} | 2D estimates only']
    if point.crease_to_knee_top_delta is not None:
        from .depth import depth_status
        dy=point.crease_to_knee_top_delta * h
        label={'crease_below_knee_top':'BELOW',
               'crease_above_knee_top':'ABOVE',
               'near_level_uncertain':'NEAR LEVEL'}[depth_status(point.crease_to_knee_top_delta)]
        rows.append(f'Crease-knee top: {dy:+.1f}px | {label}*')
    else:
        rows.append('Crease-knee top: NOT CALIBRATED')
    if event_label:
        rows.append(f'EVENT: {event_label.replace("_"," ").upper()}')
    for i,t in enumerate(rows):
        font_scale=min(.62,max(.42,w/700))
        color=(90,230,255) if event_label and i==len(rows)-1 else TEXT
        cv2.putText(result,t,(padding,24+i*30),cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(result,t,(padding,24+i*30),cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,color,1,cv2.LINE_AA)
    # Legend uses the same colours as the angle measurement guides.
    if h>200:
        y=h-17
        cv2.putText(result,'-- hip angle', (12,y),cv2.FONT_HERSHEY_SIMPLEX,.45,HIP_COLOUR,1,cv2.LINE_AA)
        cv2.putText(result,'-- knee angle',(135,y),cv2.FONT_HERSHEY_SIMPLEX,.45,KNEE_COLOUR,1,cv2.LINE_AA)
        if point.crease_point is not None:
            cv2.putText(result,'HC / KT: calibrated estimates',(12,y-20),cv2.FONT_HERSHEY_SIMPLEX,.40,DEPTH_COLOUR,1,cv2.LINE_AA)
    return result


def screenshot_name(attempt: int, event: PhaseEvent) -> str:
    stamp=f'{event.time_s:010.3f}'.replace('.','p')
    return f'attempt_{attempt:02d}_{event.name}_{stamp}s.png'

# Depth overlay uses independent anatomical reference-point *estimates*.
# NEVER label MediaPipe hip/knee joint centres as the top hip crease/knee surface.
DEPTH_COLOUR = (95, 215, 105)  # Green for calibrated depth measurement
DEPTH_REFERENCE_COLOUR = (185, 180, 245)


def draw_depth_guides(image: np.ndarray, frame: FrameMetrics):
    """Dashed horizontal height guides, an offset vertical gap and hollow point rings.

    Keep central pixels of the anatomical reference estimates visible. The
    displayed positions are tracked by a thigh-anchored calibration, not
    re-detected anatomy.
    """
    if frame.crease_point is None or frame.knee_top_point is None:
        return
    height,width=image.shape[:2]
    crease=(round(frame.crease_point[0]*width),round(frame.crease_point[1]*height))
    knee_top=(round(frame.knee_top_point[0]*width),round(frame.knee_top_point[1]*height))
    # Position the measurement bracket to the side, not over the leg.
    bracket_x=max(15,min(width-16,max(crease[0],knee_top[0])+33))
    for point,colour in ((crease,DEPTH_COLOUR),(knee_top,DEPTH_REFERENCE_COLOUR)):
        # Hollow circle and side tick: no filled dot obscuring the surface.
        cv2.circle(image,point,8,(0,0,0),2,cv2.LINE_AA)
        cv2.circle(image,point,7,colour,1,cv2.LINE_AA)
        # Start just beyond the marker rather than drawing across the surface.
        direction=1 if bracket_x>point[0] else -1
        pos=point[0]+direction*13
        while (pos<bracket_x-4 if direction==1 else pos>bracket_x+4):
            nxt=pos+direction*5
            if direction==1:
                nxt=min(nxt,bracket_x-4)
            else:
                nxt=max(nxt,bracket_x+4)
            cv2.line(image,(int(pos),point[1]),(int(nxt),point[1]),colour,2,cv2.LINE_AA)
            pos=nxt+direction*6
    if abs(crease[1]-knee_top[1])>=3:
        # Vertical distance measured separately from joint-angle rays.
        y0,y1=sorted((crease[1],knee_top[1]))
        for y in range(y0,y1,11):
            cv2.line(image,(bracket_x,y),(bracket_x,min(y+5,y1)),DEPTH_COLOUR,2,cv2.LINE_AA)
    # Short callouts close to points; not on top of the surface being evaluated.
    for label,point,colour in (('HC',crease,DEPTH_COLOUR),('KT',knee_top,DEPTH_REFERENCE_COLOUR)):
        lx=max(2,min(width-45,point[0]-33))
        ly=max(18,min(height-16,point[1]-12))
        cv2.putText(image,label,(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(image,label,(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,.42,colour,1,cv2.LINE_AA)
