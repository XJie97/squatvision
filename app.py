"""SquatVision v3: one-person, fixed-camera, offline video squat analysis.

Run: python download_model.py
     python app.py --input data/squat.mp4 --output outputs/demo
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np
from squatvision.analysis import Analysis, Config, FrameMetrics, analyse
from squatvision.geometry import best_side, extract_frame
from squatvision.depth import DepthCalibration, classify_depth, depth_status
from squatvision.optical_flow import track_anatomy
from squatvision.diagnostics import export_plots
from squatvision.render import overlay, screenshot_name


def export_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(video: Path, output: Path, model: Path, side: str, config: Config,
        min_visibility: float, start_time: float, end_time: float | None,
        no_video: bool, depth_calibration: Path | None = None) -> None:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit('MediaPipe unavailable. Install requirements in Python 3.11/3.12.') from exc
    if not video.is_file():
        raise SystemExit(f'Video not found: {video}')
    if not model.is_file():
        raise SystemExit('Pose model missing: run python download_model.py')
    calibration = DepthCalibration.load(depth_calibration) if depth_calibration is not None else None
    if calibration and side != 'auto' and side != calibration.side:
        raise SystemExit(f'--side {side} conflicts with calibrated side {calibration.side}.')
    if start_time < 0 or (end_time is not None and end_time <= start_time):
        raise SystemExit('Invalid time window. --end-time must be greater than --start-time.')
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f'Cannot decode video: {video}')
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(fps) or fps<=0 or width<1 or height<1:
        cap.release()
        raise SystemExit('Invalid FPS or video dimensions; convert to constant-frame-rate MP4.')
    if calibration and (width!=calibration.source_width or height!=calibration.source_height):
        cap.release()
        raise SystemExit('Input video dimensions differ from depth calibration; recalibrate.')
    frames:list[FrameMetrics]=[]
    selected_side=calibration.side if calibration else (None if side=='auto' else side)
    options=mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.55,
        min_pose_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    t0=time.perf_counter()
    model_s=0.0
    frame_index=0
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame=cap.read()
                if not ok:
                    break
                # Presentation timestamps are taken from frame index/fps for CFR videos.
                # MediaPipe gets monotonically increasing VIDEO-mode timestamps.
                t=frame_index/fps
                if end_time is not None and t>end_time:
                    break
                if t<start_time:
                    frame_index+=1
                    continue
                rgb=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                infer_t=time.perf_counter()
                result=landmarker.detect_for_video(image,round(t*1000))
                model_s+=time.perf_counter()-infer_t
                if result.pose_landmarks:
                    landmarks=result.pose_landmarks[0]
                    if selected_side is None:
                        selected_side=best_side(landmarks,min_visibility)
                    point=(extract_frame(landmarks,frame_index,t,width,height,
                                         selected_side,min_visibility)
                           if selected_side else FrameMetrics(frame_index,t))
                else:
                    point=FrameMetrics(frame_index,t,side=selected_side)
                frames.append(point)
                frame_index+=1
                if len(frames)%150==0:
                    print(f'Analysed {len(frames)} frames...',flush=True)
    finally:
        cap.release()
    elapsed=time.perf_counter()-t0
    if not frames:
        raise SystemExit('No frames selected for analysis.')
    if calibration is not None:
        frames=track_anatomy(video,frames,calibration,width,height)
    result=analyse(frames,config)
    output.mkdir(parents=True,exist_ok=True)
    shots=output/'event_screenshots'
    shots.mkdir(exist_ok=True)
    index_by_frame={p.frame:p for p in frames}
    events=[]
    smoothed_by_frame={f.frame:result.smoothed_crease_depth[i] for i,f in enumerate(frames)}
    # Compute high-confidence depth maxima independently of knee-angle rep phases.
    rep_depth={}
    for att in result.attempts:
        candidates=[f for f in frames if att.start_s<=f.time_s<=att.end_s
                    and f.crease_to_knee_top_delta is not None
                    and min(f.hip_crease_tracking_confidence or 0,
                            f.knee_top_tracking_confidence or 0)>=.5]
        if candidates:
            max_frame=max(candidates,key=lambda f:f.crease_to_knee_top_delta)
            thigh_length=(np.linalg.norm(np.asarray(max_frame.points[1])*[width,height]-
                                         np.asarray(max_frame.points[2])*[width,height])
                          if max_frame.points else None)
            conf=min(max_frame.hip_crease_tracking_confidence,max_frame.knee_top_tracking_confidence)
            status=classify_depth(max_frame.crease_to_knee_top_delta*height,thigh_length,conf)
            rep_depth[att.number]={'max_depth_px':round(max_frame.crease_to_knee_top_delta*height,2),
                    'max_depth_normalized':round(max_frame.crease_to_knee_top_delta*height/thigh_length,4) if thigh_length else None,
                    'max_depth_frame':max_frame.frame,'max_depth_time_s':round(max_frame.time_s,3),
                    'max_depth_hip_crease_xy':max_frame.crease_point,
                    'max_depth_knee_top_xy':max_frame.knee_top_point,
                    'max_depth_status':status,'max_depth_tracking_confidence':round(conf,3),
                    'depth_unavailable_reason':''}
        else:
            rep_depth[att.number]={'max_depth_px':None,'max_depth_normalized':None,
                    'max_depth_frame':None,'max_depth_time_s':None,
                    'max_depth_hip_crease_xy':None,'max_depth_knee_top_xy':None,
                    'max_depth_status':'unavailable','max_depth_tracking_confidence':0,
                    'depth_unavailable_reason':'no_confident_anatomical_tracking'}
    for att in result.attempts:
        for e in att.events:
            event_frame=index_by_frame[e.frame]
            smoothed_delta=smoothed_by_frame[e.frame]
            events.append({'attempt':att.number,'outcome':att.outcome,
                           'event':e.name,'timestamp_s':e.time_s,
                           'timestamp_hms':f'{int(e.time_s//60):02d}:{e.time_s%60:06.3f}',
                           'frame':e.frame,'image':str(Path('event_screenshots')/screenshot_name(att.number,e)),
                           'crease_y':event_frame.crease_point[1] if event_frame.crease_point else None,
                           'knee_top_y':event_frame.knee_top_point[1] if event_frame.knee_top_point else None,
                           'crease_minus_knee_top_px':round(event_frame.crease_to_knee_top_delta*height,2)
                                   if event_frame.crease_to_knee_top_delta is not None else None,
                           'crease_minus_knee_top_norm':event_frame.crease_to_knee_top_delta,
                           'crease_depth_status':classify_depth(smoothed_delta*height if smoothed_delta is not None else None,
                               np.linalg.norm((np.asarray(event_frame.points[1])-np.asarray(event_frame.points[2]))*[width,height])
                               if event_frame.points else None,
                               min(event_frame.hip_crease_tracking_confidence or 0,event_frame.knee_top_tracking_confidence or 0)),
                           'smoothed_crease_minus_knee_top_norm':smoothed_delta,
                           'depth_method':event_frame.depth_tracking_method,
                           'hip_crease_x':event_frame.crease_point[0] if event_frame.crease_point else None,
                           'knee_top_x':event_frame.knee_top_point[0] if event_frame.knee_top_point else None,
                           'hip_crease_tracking_confidence':event_frame.hip_crease_tracking_confidence,
                           'knee_top_tracking_confidence':event_frame.knee_top_tracking_confidence,
                           'depth_tracking_reason':event_frame.depth_unavailable_reason})
    by_frame:dict[int,list[dict]]={}
    for e in events:
        by_frame.setdefault(e['frame'],[]).append(e)
    # Re-open original video and export annotated screenshots at exact event frames.
    render_cap=cv2.VideoCapture(str(video))
    writer=None
    if not no_video:
        writer=cv2.VideoWriter(str(output/'annotated.mp4'),
                               cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
        if not writer.isOpened():
            render_cap.release()
            raise SystemExit('OpenCV could not create MP4; use --no-video to export screenshots only.')
    try:
        render_cap.set(cv2.CAP_PROP_POS_FRAMES,frames[0].frame)
        for item in frames:
            ok, raw=render_cap.read()
            if not ok:
                raise RuntimeError('Unexpected end while re-reading source video')
            frame_events=by_frame.get(item.frame,[])
            label=frame_events[0]['event'] if frame_events else None
            annotated=overlay(raw,item,result,label)
            if writer is not None:
                writer.write(annotated)
            for e in frame_events:
                # Separate screenshots for multiple events sharing a single frame.
                image=annotated if e['event']==label else overlay(raw,item,result,e['event'])
                path=output/e['image']
                if not cv2.imwrite(str(path),image):
                    raise RuntimeError(f'Cannot save event screenshot: {path}')
    finally:
        render_cap.release()
        if writer is not None:
            writer.release()
    per_frame=[]
    for i,p in enumerate(frames):
        per_frame.append({
            'frame':p.frame,'time_s':round(p.time_s,3),
            'knee_deg':p.knee_deg,'hip_deg':p.hip_deg,'hip_y':p.hip_y,
            'knee_y':p.knee_y,'visibility':p.visibility,'side':p.side,
            'smoothed_knee_deg':result.smoothed_knee[i],
            'smoothed_hip_y':result.smoothed_hip_y[i],
            'crease_x':p.crease_point[0] if p.crease_point else None,
            'crease_y':p.crease_point[1] if p.crease_point else None,
            'knee_top_x':p.knee_top_point[0] if p.knee_top_point else None,
            'knee_top_y':p.knee_top_point[1] if p.knee_top_point else None,
            'crease_minus_knee_top_norm':p.crease_to_knee_top_delta,
            'crease_depth_status':classify_depth(result.smoothed_crease_depth[i]*height
                if result.smoothed_crease_depth[i] is not None else None,
                np.linalg.norm((np.asarray(p.points[1])-np.asarray(p.points[2]))*[width,height])
                if p.points else None,
                min(p.hip_crease_tracking_confidence or 0,p.knee_top_tracking_confidence or 0)),
            'smoothed_crease_minus_knee_top_norm':result.smoothed_crease_depth[i],
            'timestamp_s':p.time_s,
            'hip_crease_x':p.crease_point[0]*width if p.crease_point else None,
            'hip_crease_y':p.crease_point[1]*height if p.crease_point else None,
            'knee_top_x_px':p.knee_top_point[0]*width if p.knee_top_point else None,
            'knee_top_y_px':p.knee_top_point[1]*height if p.knee_top_point else None,
            'depth_difference_px':p.crease_to_knee_top_delta*height if p.crease_to_knee_top_delta is not None else None,
            'depth_difference_normalized':(p.crease_to_knee_top_delta*height /
                np.linalg.norm((np.asarray(p.points[1])-np.asarray(p.points[2]))*[width,height]))
                if p.crease_to_knee_top_delta is not None and p.points and
                   np.linalg.norm((np.asarray(p.points[1])-np.asarray(p.points[2]))*[width,height])>0 else None,
            'depth_status':classify_depth(p.crease_to_knee_top_delta*height if p.crease_to_knee_top_delta is not None else None,
                np.linalg.norm((np.asarray(p.points[1])-np.asarray(p.points[2]))*[width,height]) if p.points else None,
                min(p.hip_crease_tracking_confidence or 0,p.knee_top_tracking_confidence or 0)),
            'hip_crease_tracking_confidence':p.hip_crease_tracking_confidence,
            'knee_top_tracking_confidence':p.knee_top_tracking_confidence,
            'depth_tracking_method':p.depth_tracking_method,
            'depth_unavailable_reason':p.depth_unavailable_reason,
        })
    export_csv(output/'frames.csv',per_frame,list(per_frame[0]))
    attempt_rows=[]
    for a in result.attempts:
        attempt_rows.append({**{k:v for k,v in a.as_dict().items() if k!='events'},
                             **rep_depth[a.number]})
    attempt_fields=list(attempt_rows[0]) if attempt_rows else [
        'number','outcome','start_s','end_s','duration_s','knee_min_deg',
        'hip_min_deg','hip_drop','hip_below_knee_proxy','bottom_pause_s',
        'max_crease_depth_delta','crease_below_knee_top_observed','note',
        'max_depth_px','max_depth_normalized','max_depth_frame','max_depth_time_s',
        'max_depth_hip_crease_xy','max_depth_knee_top_xy','max_depth_status',
        'max_depth_tracking_confidence','depth_unavailable_reason']
    export_csv(output/'attempts.csv',attempt_rows,attempt_fields)
    export_csv(output/'events.csv',events,
               ['attempt','outcome','event','timestamp_s','timestamp_hms','frame','image',
                'crease_y','knee_top_y','crease_minus_knee_top_px',
                'crease_minus_knee_top_norm','crease_depth_status','depth_method',
                'smoothed_crease_minus_knee_top_norm','hip_crease_x','knee_top_x',
                'hip_crease_tracking_confidence','knee_top_tracking_confidence',
                'depth_tracking_reason'])
    summary={
        'source':video.name,'resolution':[width,height],'fps':fps,
        'analysed_frames':len(frames),'selected_side':selected_side,
        'usable_pose_fraction':round(sum(p.knee_deg is not None for p in frames)/len(frames),4),
        'upright_knee_reference_deg':result.upright_knee_deg,
        'upright_hip_reference_y':result.upright_hip_y,
        'completed_reps':len(result.completed),'total_attempts':len(result.attempts),
        'incomplete_attempts':sum(a.outcome=='incomplete' for a in result.attempts),
        'suspected_failures':sum(a.outcome=='suspected_failure' for a in result.attempts),
        'rejected_cycles':result.candidate_cycles_rejected,
        'inference_ms_per_frame':round(model_s/len(frames)*1000,2),
        'offline_processing_fps':round(len(frames)/elapsed,2),
        'configuration':config.__dict__,
        'depth_calibration_applied':calibration is not None,
        'depth_calibration_reference_s':calibration.reference_time_s if calibration else None,
        'depth_method':calibration.method if calibration else 'not_calibrated',
        'depth_metric_note':'Manually clicked anatomical surface references are independently tracked via local visual flow; 2D approximate, not validated judging.',
        'per_attempt_depth':rep_depth,
        'warning':result.warning,
        'limitations':'2D view-dependent heuristics; NOT competition-validated depth or failure detection.',
    }
    if calibration is not None:
        export_plots(frames,output/'diagnostics',height,width)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (output/'events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    for a in result.attempts:
        print(f'Attempt {a.number}: {a.outcome}, knee minimum {a.knee_min_deg:.1f} deg, '
              f'hip drop {a.hip_drop:.3f}, bottom pause ~{a.bottom_pause_s:.2f}s, '
              f'crease-depth observed={a.crease_below_knee_top_observed}')
        for e in a.events:
            print(f'  {e.name}: {e.time_s:.3f}s [frame {e.frame}]')
    print('Saved:',output.resolve())


def main():
    p=argparse.ArgumentParser(description='SquatVision v3: squat attempts, phases and calibrated depth guides')
    p.add_argument('--input',required=True,type=Path,help='Input recorded, constant-frame-rate MP4')
    p.add_argument('--output',type=Path,default=Path('outputs/demo'))
    p.add_argument('--model',type=Path,default=Path('models/pose_landmarker_lite.task'))
    p.add_argument('--side',choices=['auto','left','right'],default='auto')
    p.add_argument('--min-visibility',type=float,default=0.55)
    p.add_argument('--start-time',type=float,default=0.,help='Analyse only after timestamp in seconds')
    p.add_argument('--end-time',type=float,default=None,help='Stop at timestamp in seconds')
    p.add_argument('--min-knee-excursion',type=float,default=40.,help='Required knee ROM in degrees')
    p.add_argument('--min-hip-drop',type=float,default=.075,help='Required hip drop, fraction of frame height')
    p.add_argument('--upright-hold',type=float,default=.30,help='Standing hold in seconds')
    p.add_argument('--lockout-hold',type=float,default=.35,help='Return-to-upright hold in seconds')
    p.add_argument('--no-video',action='store_true',help='Export frames and event screenshots but skip MP4 rendering')
    p.add_argument('--depth-calibration',type=Path,default=None,
                   help='Path to JSON created by calibrate_depth.py; required for anatomical depth tracking')
    args=p.parse_args()
    if not 0<=args.min_visibility<=1:
        p.error('--min-visibility must be in [0,1]')
    if not 0<args.min_knee_excursion<150 or not 0<args.min_hip_drop<1:
        p.error('Minimum knee excursion or hip drop is out of range')
    config=Config(min_knee_excursion_deg=args.min_knee_excursion,
                  min_hip_drop=args.min_hip_drop,upright_hold_s=args.upright_hold,
                  return_hold_s=args.lockout_hold)
    run(args.input,args.output,args.model,args.side,config,
        args.min_visibility,args.start_time,args.end_time,args.no_video,args.depth_calibration)


if __name__=='__main__':
    main()
