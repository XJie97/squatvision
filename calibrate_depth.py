"""Manually locate hip crease and top of knee on a sample video frame.

Usage:
  python calibrate_depth.py --input data/squat.mp4 --time 39.2 \
      --side right --output depth_calibration.json

For headless systems, supply --hip-crease X,Y --knee-top X,Y in ORIGINAL pixels.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np

from squatvision.geometry import best_side, extract_frame
from squatvision.depth import DepthCalibration, apply_depth_calibration
from squatvision.render import draw_depth_guides


def read_frame(video: Path, seconds: float):
    cap=cv2.VideoCapture(str(video))
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


def choose_points(image: np.ndarray) -> tuple[tuple[float,float],tuple[float,float]]:
    """GUI: click top hip crease then top knee joint surface; Enter to save."""
    h,w=image.shape[:2]
    scale=min(1.0,1200/max(w,1),780/max(h,1))
    size=(max(1,int(round(w*scale))),max(1,int(round(h*scale))))
    selected=[]
    title='Depth calibration | 1: hip crease, 2: knee top | Enter=save, R=reset, Esc=cancel'
    try:
        cv2.namedWindow(title,cv2.WINDOW_AUTOSIZE)
    except cv2.error as exc:
        raise SystemExit('OpenCV GUI unavailable; supply --hip-crease and --knee-top coordinates.') from exc

    def callback(event,x,y,_flags,_userdata):
        if event==cv2.EVENT_LBUTTONDOWN and len(selected)<2:
            selected.append((max(0,min(w-1,round(x/scale))),max(0,min(h-1,round(y/scale)))))
    cv2.setMouseCallback(title,callback)
    try:
        while True:
            display=cv2.resize(image,size)
            labels=['HIP CREASE','TOP OF KNEE']
            for i,p in enumerate(selected):
                loc=(int(round(p[0]*scale)),int(round(p[1]*scale)))
                cv2.circle(display,loc,7,(0,250,240) if i==0 else (245,170,250),2)
                cv2.putText(display,labels[i],(min(loc[0]+12,size[0]-110),max(24,loc[1]-11)),
                            cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(display,labels[i],(min(loc[0]+12,size[0]-110),max(24,loc[1]-11)),
                            cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            label=('Click visible top of hip crease' if len(selected)==0 else
                   'Click top surface of the knee joint' if len(selected)==1 else
                   'Enter: confirm | R: reselect')
            cv2.rectangle(display,(0,0),(size[0],36),(0,0,0),-1)
            cv2.putText(display,label,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.59,(245,245,245),2)
            cv2.imshow(title,display)
            key=cv2.waitKey(40)&255
            if key in (13,10) and len(selected)==2:
                return selected[0],selected[1]
            if key in (ord('r'),ord('R')):
                selected.clear()
            if key==27:
                raise SystemExit('Calibration cancelled.')
    finally:
        cv2.destroyWindow(title)


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
    crease,top=(args.hip_crease,args.knee_top) if args.hip_crease else choose_points(raw)
    calib=DepthCalibration.from_reference(pose,crease,top,raw.shape[1],raw.shape[0])
    calib.save(args.output)
    estimated=apply_depth_calibration(pose,calib,raw.shape[1],raw.shape[0])
    preview=raw.copy()
    draw_depth_guides(preview,estimated)
    args.preview.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(args.preview),preview):
        raise SystemExit(f'Failed to save calibration preview: {args.preview}')
    print(f'Calibration preview: {args.preview}')
    print(f'Saved {args.output} for side={calib.side}; reference {t:.3f}s frame={index}.')
    print('Reference points are propagated by thigh pose. These are estimated surfaces, not re-detected landmarks.')


if __name__=='__main__':
    main()
