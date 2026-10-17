"""SquatVision v5: one-person, fixed-camera, offline video squat analysis.

Run: python download_model.py
     python app.py --input data/squat.mp4 --output outputs/demo
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
from dataclasses import replace
from collections import Counter
import cv2
import numpy as np
from squatvision.analysis import Analysis, Config, FrameMetrics, analyse
from squatvision.geometry import best_side, extract_frame
from squatvision.depth import DepthCalibration, classify_depth, depth_status
from squatvision.optical_flow import track_anatomy, calibration_from_manual_annotations
from squatvision.diagnostics import export_plots
from squatvision.render import overlay, screenshot_name
from squatvision.batch import BatchInputError, run_batch
from squatvision.ipf_depth import load_manual, apply_manual, measure, attempt_depth, UNAVAILABLE, MANUAL, VERIFIED


def export_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(video: Path, output: Path, model: Path, side: str, config: Config,
        min_visibility: float, start_time: float, end_time: float | None,
        no_video: bool, depth_calibration: Path | None = None,
        manual_annotations: Path | None = None,
        manual_only: bool = False) -> None:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit('MediaPipe unavailable. Install requirements in Python 3.11/3.12.') from exc
    if manual_only and depth_calibration is not None:
        raise SystemExit('--manual-only cannot be combined with --depth-calibration.')
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
    manual_points=(load_manual(manual_annotations,video.name,width,height,fps)
                   if manual_annotations is not None else {})
    # A manual annotation is a verified image point at ONE frame, not a video
    # track. When no separate depth calibration is supplied, automatically
    # initialise optical flow from suitable manually labelled pose frames.
    # Failure to follow a point remains UNAVAILABLE, never interpolated or
    # silently substituted with the MediaPipe joint centre.
    manual_tracking_warning = ''
    manual_seeded_tracking = False
    if calibration is None and manual_points and not manual_only:
        calibration, manual_tracking_warning = calibration_from_manual_annotations(
            frames, manual_points, width, height)
        manual_seeded_tracking = calibration is not None
        if manual_tracking_warning:
            print('Manual anchor tracking:', manual_tracking_warning, flush=True)
    elif calibration is not None and manual_points and not manual_only:
        existing={calibration.reference_frame,*[a[0] for a in calibration.additional_references]}
        extra=list(calibration.additional_references)
        lookup={f.frame:f for f in frames}
        for i,label in sorted(manual_points.items()):
            f=lookup.get(i)
            if (i not in existing and f is not None and
                    f.side==calibration.side and f.points and f.knee_deg is not None):
                hx,hy=label.upper_thigh_at_hip;kx,ky=label.top_of_knee
                extra.append((i,hx,hy,kx,ky))
                existing.add(i)
        if len(extra)!=len(calibration.additional_references):
            calibration=replace(calibration,additional_references=tuple(sorted(extra)))
            calibration.validate()
    if calibration is not None:
        frames=track_anatomy(video,frames,calibration,width,height)
    if manual_points:
        frames=[apply_manual(f,manual_points[f.frame],width,height)
                if f.frame in manual_points else f for f in frames]
    # Distinct data flows: joint centre kinetics -> repetition analysis;
    # anatomical SURFACE tracking/manual annotations -> IPF relative height.
    measurements=[measure(f,width,height,require_manual_verification=True) for f in frames]
    measurement_by_frame={m.frame:m for m in measurements}
    result=analyse(frames,config)
    output.mkdir(parents=True,exist_ok=True)
    shots=output/'event_screenshots'
    shots.mkdir(exist_ok=True)
    index_by_frame={p.frame:p for p in frames}
    events=[]
    smoothed_by_frame={f.frame:result.smoothed_crease_depth[i] for i,f in enumerate(frames)}
    # Peak image-plane surface depth is selected independently of knee angles.
    # Missing tracked frames are never filled with joint-centre measurements.
    rep_assessments={a.number:attempt_depth(
        measurements,a.start_s,a.end_s,a.number,a.outcome=='complete',
        bottom_time_s=next((e.time_s for e in a.events if e.name=='descent_end'),None))
        for a in result.attempts}
    rep_depth={}
    for a in result.attempts:
        d=rep_assessments[a.number]['depth_assessment']
        rep_depth[a.number]={
            'max_depth_px':d['depth_margin_px'],
            'max_depth_normalized':d['depth_margin_normalized'],
            'max_depth_frame':d['maximum_depth_frame'],
            'max_depth_time_s':d['maximum_depth_timestamp_s'],
            'max_depth_hip_crease_xy':d.get('upper_thigh_at_hip_px'),
            'max_depth_knee_top_xy':d.get('top_of_knee_px'),
            'max_depth_status':d['status'].lower(),
            'max_depth_tracking_confidence':d.get('tracking_quality'),
            'depth_unavailable_reason':'' if d['status']!=UNAVAILABLE else 'insufficient_surface_observations',
            'depth_assessment':d,
        }
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
                           'crease_depth_status':measurement_by_frame[e.frame].depth_status.lower(),
                           'smoothed_crease_minus_knee_top_norm':smoothed_delta,
                           'depth_method':event_frame.depth_tracking_method,
                           'hip_crease_x':event_frame.crease_point[0] if event_frame.crease_point else None,
                           'knee_top_x':event_frame.knee_top_point[0] if event_frame.knee_top_point else None,
                           'hip_crease_tracking_confidence':event_frame.hip_crease_tracking_confidence,
                           'knee_top_tracking_confidence':event_frame.knee_top_tracking_confidence,
                           'depth_tracking_reason':event_frame.depth_unavailable_reason,
                           'measurement_source':measurement_by_frame[e.frame].measurement_source,
                           'depth_uncertainty_px':measurement_by_frame[e.frame].depth_uncertainty_px})
    # Add timestamped screenshots for manually annotated frames and peak depths.
    extra_shots={}
    for frame_id in manual_points:
        if frame_id in index_by_frame:
            extra_shots[frame_id]=f'manual_depth_{frame_id:06d}.png'
    for d in rep_depth.values():
        if d['max_depth_frame'] is not None:
            i=d['max_depth_frame'];extra_shots.setdefault(i,f'peak_depth_{i:06d}.png')
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
            if item.frame in extra_shots:
                name=extra_shots[item.frame]
                folder=output/'depth_screenshots';folder.mkdir(exist_ok=True)
                if not cv2.imwrite(str(folder/name),annotated):
                    raise RuntimeError(f'Cannot write depth screenshot: {name}')
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
        m=measurements[i]
        per_frame.append({**m.to_dict(),
            'frame':p.frame,'time_s':round(p.time_s,3),
            'knee_deg':p.knee_deg,'hip_deg':p.hip_deg,'hip_y':p.hip_y,
            'knee_y':p.knee_y,'visibility':p.visibility,'side':p.side,
            'smoothed_knee_deg':result.smoothed_knee[i],
            'smoothed_hip_y':result.smoothed_hip_y[i],
            'crease_x':p.crease_point[0] if p.crease_point else None,
            'crease_y':p.crease_point[1] if p.crease_point else None,
            'knee_top_x':p.knee_top_point[0] if p.knee_top_point else None,
            'knee_top_surface_x_px':m.knee_top_x,
            'knee_top_y':p.knee_top_point[1] if p.knee_top_point else None,
            'knee_top_surface_y_px':m.knee_top_y,
            'crease_minus_knee_top_norm':p.crease_to_knee_top_delta,
            'crease_depth_status':m.depth_status.lower(),
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
            'depth_status':m.depth_status.lower(),
            'ipf_depth_status':m.depth_status,
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
                'depth_tracking_reason','measurement_source','depth_uncertainty_px'])
    # Diagnose the actual squat bottom, rather than a misleading all-video
    # percentage. Both anatomical surface points must be present for a depth
    # observation; a reliable MediaPipe *joint* angle is not sufficient.
    bottom_tracking_coverage={}
    for attempt in result.attempts:
        bottom_event=next((e for e in attempt.events if e.name=='descent_end'),None)
        if bottom_event is None:
            continue
        region=[p for p in frames if abs(p.time_s-bottom_event.time_s)<=.25]
        available=[p for p in region if p.crease_point is not None and p.knee_top_point is not None]
        bottom_tracking_coverage[str(attempt.number)]={
            'bottom_time_s':round(bottom_event.time_s,3),
            'window_s':0.25,
            'frames_in_window':len(region),
            'depth_measurement_frames':len(available),
            'manual_frames':sum(p.depth_tracking_method in ('MANUAL_ANNOTATION','MANUAL_VERIFIED_TRACKING') for p in available),
            'optical_flow_frames':sum(p.depth_tracking_method=='local_optical_flow' for p in available),
            'bottom_frame_has_depth':any(p.frame==bottom_event.frame for p in available),
            'missing_reasons':dict(Counter(
                p.depth_unavailable_reason or 'no_anatomical_surface_estimate'
                for p in region if p.crease_point is None or p.knee_top_point is None)),
        }
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
        'depth_calibration_applied':depth_calibration is not None,
        'manual_seeded_visual_tracking':manual_seeded_tracking,
        'manual_tracking_warning':manual_tracking_warning,
        'manual_annotation_count':len(manual_points),
        'anatomical_tracking_valid_frames':sum(
            p.crease_point is not None and p.knee_top_point is not None and
            p.depth_tracking_method=='local_optical_flow' for p in frames),
        'anatomical_tracking_unavailable_reasons':dict(Counter(
            p.depth_unavailable_reason for p in frames
            if p.depth_tracking_method=='local_optical_flow' and p.crease_point is None)),
        'bottom_tracking_coverage':bottom_tracking_coverage,
        'manual_depth_annotations_applied':bool(manual_points),
        'depth_calibration_reference_s':calibration.reference_time_s if calibration else None,
        'depth_method':('manual_annotation_seeded_optical_flow' if manual_seeded_tracking
                        else calibration.method if calibration else 'manual_frames_only' if manual_points
                        else 'not_calibrated'),
        'depth_metric_note':'Optical-flow margins are provisional and not classified as above/below without manual verification. A missing bottom cannot establish insufficient depth. Signed image-plane pixels, not cm. No official IPF ruling.',
        'per_attempt_depth':rep_depth,
        'ipf_depth_assessments':list(rep_assessments.values()),
        'warning':result.warning,
        'limitations':'2D view-dependent heuristics; NOT competition-validated depth or failure detection.',
    }
    if calibration is not None or manual_points:
        export_plots(frames,output/'diagnostics',height,width)
        from squatvision.ipf_plot import plot_depth
        plot_depth(measurements,output/'depth_vs_time.svg')
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
    p=argparse.ArgumentParser(description='SquatVision v5: squat phases and experimental IPF surface-depth assessment')
    sources=p.add_mutually_exclusive_group()
    sources.add_argument('--input',type=Path,default=None,
                         help='Analyse one recorded MP4 (legacy mode)')
    sources.add_argument('--input-dir',type=Path,default=None,
                         help='Analyse every .mp4 in directory (default: data)')
    p.add_argument('--output',type=Path,default=None,
                   help='Single-video destination or batch output root (default: outputs)')
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
    p.add_argument('--manual-annotations',type=Path,default=None,
                   help='Manual depth annotations JSON; also seeds provisional optical flow between labelled frames')
    p.add_argument('--annotations-dir',type=Path,default=None,
                   help='Batch: per-video <stem>.json manual annotation files')
    p.add_argument('--calibrations-dir',type=Path,default=None,
                   help='Batch: per-video <stem>.json calibration files')
    p.add_argument('--manual-only',action='store_true',
                   help='Disable visual tracking between manually marked frames')
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
    if args.input is not None:
        if args.annotations_dir is not None or args.calibrations_dir is not None:
            p.error('--annotations-dir and --calibrations-dir require batch mode.')
        run(args.input, args.output or Path('outputs/demo'), args.model, args.side, config,
            args.min_visibility, args.start_time, args.end_time, args.no_video,
            args.depth_calibration, args.manual_annotations, args.manual_only)
        return
    # Without --input, batch process all MP4s directly inside data/.
    # Separate folders and per-video annotation files prevent camera mixing.
    if args.manual_annotations is not None or args.depth_calibration is not None:
        p.error('For batch processing, use --annotations-dir/--calibrations-dir '
                'containing separate <stem>.json files instead.')
    batch_input = args.input_dir or Path('data')
    batch_output = args.output or Path('outputs')
    try:
        rows = run_batch(
            batch_input, batch_output,
            process_video=lambda video, dest, manual, calibration: run(
                video, dest, args.model, args.side, config,
                args.min_visibility, args.start_time, args.end_time, args.no_video,
                calibration, manual, args.manual_only),
            annotations_dir=args.annotations_dir,
            calibrations_dir=args.calibrations_dir,
        )
    except BatchInputError as exc:
        p.error(str(exc))
    if any(row['status'] == 'failed' for row in rows):
        raise SystemExit(1)


if __name__=='__main__':
    main()
