"""Manually locate hip crease and top of knee on a sample video frame.

Usage:
  python calibrate_depth.py --input data/squat.mp4 --time 39.2 \
      --side right --output depth_calibration.json

For headless systems, supply --hip-crease X,Y --knee-top X,Y in ORIGINAL pixels.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path

import cv2
from squatvision.video import open_video
import numpy as np

from squatvision.geometry import best_side, extract_frame
from squatvision.depth import DepthCalibration, apply_depth_calibration
from squatvision.optical_flow import LocalPointTracker
from squatvision.render import draw_depth_guides


def read_frame(video: Path, seconds: float):
    cap=open_video(str(video))
    if not cap.isOpened():
        raise SystemExit(f'Cannot read video: {video}')
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps<=0:
        cap.release()
        raise SystemExit('Invalid FPS. Use constant-frame-rate footage.')
    i=round(seconds*fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES,i)
    ok,image=cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f'Could not read frame at {seconds:.3f}s')
    return image,i,i/fps


def infer_pose(image, model: Path, index: int, timestamp: float, side: str):
    try:
        import mediapipe as mp
    except ImportError as ex:
        raise SystemExit('Install MediaPipe with: python -m pip install -r requirements.txt') from ex
    rgb=np.ascontiguousarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    options=mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        result=landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
    if not result.pose_landmarks:
        raise SystemExit('No person detected in reference frame; select another time.')
    points=result.pose_landmarks[0]
    chosen=best_side(points,0.55) if side=='auto' else side
    if not chosen:
        raise SystemExit('No reliably visible side; choose a different reference frame.')
    metric=extract_frame(points,index,timestamp,image.shape[1],image.shape[0],chosen,0.55)
    if not metric.points:
        raise SystemExit(f'Side {chosen} not sufficiently visible; change --side or time.')
    return metric


def _tracking_preview(video: Path, cal: DepthCalibration, output: Path,
                      fps: float | None = None) -> Path:
    """Preview reference +/- 0.6s using independent optical flow, no pose fabrication."""
    capture=open_video(str(video))
    fps=fps or float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps<=0:
        capture.release()
        raise ValueError('Invalid video FPS for preview')
    ref=cal.reference_frame
    count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    def frame_at(idx):
        capture.set(cv2.CAP_PROP_POS_FRAMES,idx)
        ok,img=capture.read()
        return img if ok else None
    source=frame_at(ref)
    if source is None:
        capture.release()
        raise ValueError('Reference frame could not be decoded')
    reference_gray=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)
    points={'hip crease':cal.reference_hip_crease_px,'knee top':cal.reference_knee_top_px}
    annotated={ref:(source,tuple(points.values()),1.)}
    half=max(3,int(round(.6*fps)))
    for direction in (-1,1):
        trackers=[LocalPointTracker(reference_gray,point) for point in points.values()]
        for i in range(ref+direction,max(-1,min(count,ref+direction*(half+1))),-1 if direction<0 else 1):
            if i<0 or i>=count:break
            img=frame_at(i)
            if img is None:break
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            for tracker in trackers:
                tracker.update(gray)
            if all(t.active for t in trackers):
                annotated[i]=(img,[tuple(t.xy) for t in trackers],min(t.confidence for t in trackers))
            else:
                annotated[i]=(img,[],0.)
    capture.release()
    candidate=[ref+round(t*fps) for t in (-.6,-.4,-.2,0,.2,.4,.6)]
    tiles=[]
    w=240; h=330
    for i in candidate:
        if i not in annotated:
            tile=np.zeros((h,w,3),np.uint8)
            cv2.putText(tile,'TRACKING UNAVAILABLE',(5,90),cv2.FONT_HERSHEY_SIMPLEX,.42,(250,200,150),1)
        else:
            img,refpoints,confidence=annotated[i]
            tile=cv2.resize(img,(w,h))
            for k,p in enumerate(refpoints):
                xy=(round(p[0]*w/img.shape[1]),round(p[1]*h/img.shape[0]))
                cv2.circle(tile,xy,6,((75,215,105) if k==0 else (185,180,245)),2,cv2.LINE_AA)
            cv2.rectangle(tile,(0,0),(w,29),(8,12,18),-1)
            cv2.putText(tile,f'{i/fps:.2f}s  conf {confidence:.2f}',(5,20),
                        cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1,cv2.LINE_AA)
        tiles.append(tile)
    mosaic=np.hstack(tiles)
    output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(output),mosaic):
        raise RuntimeError('Could not save tracking preview')
    return output


def choose_points(image: np.ndarray, pose, video: Path,
                  reference_frame: int) -> tuple[tuple[float,float],tuple[float,float]]:
    """Interactive point editor with landmark offsets and 4x zoom panel.

    1/2 select marker, left click full view or zoom to reposition, +/- change
    zoom, P preview nearby frames, Enter/Space/S validate and save, R reset.
    """
    h,w=image.shape[:2]
    scale=min(2.,1120/max(w,1),760/max(h,1))
    view_size=(max(1,round(w*scale)),max(1,round(h*scale)))
    selected=[None,None]
    active=0
    mouse=[w//2,h//2]
    magnification=[4]
    title='SquatVision depth calibration'
    zoom_title='Zoom - click to reposition selected point'
    anchors=[(pose.points[1][0]*w,pose.points[1][1]*h),
             (pose.points[2][0]*w,pose.points[2][1]*h)]
    try:
        cv2.namedWindow(title,cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(zoom_title,cv2.WINDOW_AUTOSIZE)
    except cv2.error as exc:
        raise SystemExit('OpenCV GUI unavailable; use --hip-crease and --knee-top.') from exc
    zoom_crop=[0,0,1]
    def callback(event,x,y,_flags,_userdata):
        nonlocal active
        mouse[:]=[min(w-1,max(0,round(x/scale))),min(h-1,max(0,round(y/scale)))]
        if event==cv2.EVENT_LBUTTONDOWN:
            if all(p is not None for p in selected):
                near=min(range(2),key=lambda idx:np.hypot(selected[idx][0]-mouse[0],selected[idx][1]-mouse[1]))
                if np.linalg.norm(np.asarray(selected[near])-mouse)<25:
                    active=near
            selected[active]=tuple(mouse)
            if selected[1-active] is None:active=1-active
    def zoom_callback(event,x,y,_flags,_userdata):
        if event!=cv2.EVENT_LBUTTONDOWN:return
        left,top,factor=zoom_crop
        px=max(0,min(w-1,round(left+x/factor)))
        py=max(0,min(h-1,round(top+y/factor)))
        selected[active]=(px,py)
    cv2.setMouseCallback(title,callback)
    cv2.setMouseCallback(zoom_title,zoom_callback)
    note=''
    try:
        while True:
            display=cv2.resize(image,view_size)
            for idx,joint in enumerate(anchors):
                center=tuple(round(v*scale) for v in joint)
                cv2.drawMarker(display,center,(255,180,50),cv2.MARKER_CROSS,13,1)
                cv2.putText(display,'MP hip' if idx==0 else 'MP knee',
                            (max(2,center[0]-45),max(17,center[1]-12)),
                            cv2.FONT_HERSHEY_SIMPLEX,.37,(255,180,50),1)
            for idx,p in enumerate(selected):
                if p is None:continue
                center=tuple(round(v*scale) for v in p)
                colour=(65,240,110) if idx==0 else (225,175,250)
                cv2.circle(display,center,7,colour,2)
                cv2.line(display,center,tuple(round(v*scale) for v in anchors[idx]),colour,1)
                dist=np.linalg.norm(np.asarray(p)-anchors[idx])
                cv2.putText(display,('HIP' if idx==0 else 'KNEE')+f' offset {dist:.1f}px',
                            (max(3,min(center[0]+12,view_size[0]-170)),max(28,center[1]-12)),
                            cv2.FONT_HERSHEY_SIMPLEX,.43,colour,1)
            cv2.rectangle(display,(0,0),(view_size[0],46),(8,12,18),-1)
            cv2.putText(display,f'1=hip 2=knee | active={active+1} | +/- zoom | P=preview',
                        (7,16),cv2.FONT_HERSHEY_SIMPLEX,.38,(240,240,240),1)
            cv2.putText(display,note or 'Enter/Space/S save | R reset | Esc cancel',
                        (7,36),cv2.FONT_HERSHEY_SIMPLEX,.38,(240,240,240),1)
            cv2.imshow(title,display)
            focus=selected[active] if selected[active] is not None else mouse
            size=60
            left=max(0,min(w-size,round(focus[0]-size/2))) if w>size else 0
            top=max(0,min(h-size,round(focus[1]-size/2))) if h>size else 0
            crop=image[top:top+min(h,60),left:left+min(w,60)]
            zoom=cv2.resize(crop,(crop.shape[1]*magnification[0],crop.shape[0]*magnification[0]),interpolation=cv2.INTER_NEAREST)
            for idx,p in enumerate(selected):
                if p is not None and left<=p[0]<left+crop.shape[1] and top<=p[1]<top+crop.shape[0]:
                    xy=(round((p[0]-left)*magnification[0]),round((p[1]-top)*magnification[0]))
                    cv2.circle(zoom,xy,8,(65,240,110) if idx==0 else (225,175,250),2)
            zoom_crop[:]=[left,top,magnification[0]]
            cv2.imshow(zoom_title,zoom)
            key=cv2.waitKey(50)&255
            if key in (ord('1'),ord('2')):active=int(chr(key))-1
            elif key in (ord('+'),ord('=')):magnification[0]=min(8,magnification[0]+1)
            elif key in (ord('-'),ord('_')):magnification[0]=max(2,magnification[0]-1)
            elif key in (ord('r'),ord('R')):selected[:]=[None,None];active=0;note=''
            elif key in (ord('p'),ord('P')) and all(v is not None for v in selected):
                try:
                    cal=DepthCalibration.from_reference(pose,selected[0],selected[1],w,h)
                    path=_tracking_preview(video,cal,Path('outputs/depth_tracking_preview.png'))
                    note=f'Preview: {path}'
                except ValueError as exc:note=str(exc)[:65]
            elif key in (13,10,32,ord('s'),ord('S')):
                if all(p is not None for p in selected):
                    try:
                        DepthCalibration.from_reference(pose,selected[0],selected[1],w,h)
                    except ValueError as exc:
                        note=str(exc)[:65]
                        continue
                    return tuple(selected)
                note='Select both markers before saving.'
            elif key==27:
                raise SystemExit('Calibration cancelled')
    finally:
        cv2.destroyWindow(title)
        cv2.destroyWindow(zoom_title)

def _pixel_pair(value: str):
    try:
        x,y=(float(p.strip()) for p in value.split(','))
        return (x,y)
    except (ValueError,TypeError):
        raise argparse.ArgumentTypeError('Expected pixel coordinates X,Y e.g. 220,460')


def main():
    parser=argparse.ArgumentParser(description='Calibrate hip crease and knee top on source video')
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--time',type=float,required=True,help='Reference frame timestamp in seconds; choose bottom of squat')
    parser.add_argument('--model',type=Path,default=Path('models/pose_landmarker_lite.task'))
    parser.add_argument('--side',choices=['left','right','auto'],default='auto')
    parser.add_argument('--output',type=Path,default=Path('depth_calibration.json'))
    parser.add_argument('--preview',type=Path,default=Path('outputs/depth_calibration_preview.png'),
                        help='Where to save an annotated reference image for inspection')
    parser.add_argument('--append',type=Path,default=None,
        help='Add manually selected reference frame to existing v2 calibration JSON')
    parser.add_argument('--hip-crease',type=_pixel_pair,help='Headless reference-point coordinates X,Y')
    parser.add_argument('--knee-top',type=_pixel_pair,help='Headless reference-point coordinates X,Y')
    args=parser.parse_args()
    if args.time<0:
        parser.error('--time must be non-negative')
    if (args.hip_crease is None) != (args.knee_top is None):
        parser.error('Supply both --hip-crease and --knee-top, or neither')
    if not args.model.is_file():
        parser.error('Model missing: run python download_model.py')
    raw,index,t=read_frame(args.input,args.time)
    pose=infer_pose(raw,args.model,index,t,args.side)
    crease,top=(args.hip_crease,args.knee_top) if args.hip_crease else choose_points(raw,pose,args.input,index)
    current=DepthCalibration.from_reference(pose,crease,top,raw.shape[1],raw.shape[0])
    if args.append is not None:
        prior=DepthCalibration.load(args.append)
        if (prior.version!=2 or prior.side!=pose.side or
            prior.source_width!=raw.shape[1] or prior.source_height!=raw.shape[0]):
            parser.error('--append requires compatible v2 file, side and image dimensions')
        if index in [prior.reference_frame,*[item[0] for item in prior.additional_references]]:
            parser.error('This reference frame is already calibrated')
        new_row=(index,*crease,*top)
        calib=replace(prior,additional_references=(*prior.additional_references,new_row))
    else:
        calib=current
    calib.save(args.output)
    estimated=apply_depth_calibration(pose,calib,raw.shape[1],raw.shape[0])
    preview=raw.copy()
    draw_depth_guides(preview,estimated)
    args.preview.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(args.preview),preview):
        raise SystemExit(f'Failed to save calibration preview: {args.preview}')
    print(f'Calibration preview: {args.preview}')
    print(f'Saved {args.output} for side={calib.side}; reference {t:.3f}s frame={index}.')
    local_preview=args.preview.with_name('depth_tracking_preview.png')
    try:
        _tracking_preview(args.input,calib,local_preview)
        print(f'Nearby-frame tracking preview: {local_preview}')
    except (RuntimeError,ValueError) as exc:
        print(f'Nearby-frame preview unavailable: {exc}')
    print('Local visual tracking estimates these points between frames; validate motion against the visible anatomy.')


if __name__=='__main__':
    main()
