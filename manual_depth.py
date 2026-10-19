"""GUI/manual fallback for IPF anatomical-surface annotations.

This tool does not use MediaPipe or optical flow. Manual annotation may be
completed even when the automatic depth tracker never finds a valid frame.
Keyboard: A/D previous/next frame, J/L skip 10, click hip then knee, R clear,
S/Enter save, +/- change zoom, Esc/window X cancel without saving.
Mouse events redraw immediately; closed windows never reopen.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import csv
import json

import cv2
from squatvision.video import open_video
import numpy as np

from squatvision.ipf_depth import (ManualPoint, MANUAL, apply_manual, load_manual,
                                  save_manual, measure)
from squatvision.analysis import FrameMetrics
from squatvision.ipf_plot import plot_depth


def render_manual(image: np.ndarray, manual: ManualPoint | None,
                  selected: list[tuple[float,float]], frame_id: int, fps: float,
                  zoom: float=1.7) -> np.ndarray:
    h,w=image.shape[:2]
    drawing=cv2.resize(image,None,fx=zoom,fy=zoom,interpolation=cv2.INTER_LINEAR)
    if manual:
        points=[manual.upper_thigh_at_hip,manual.top_of_knee]
    else:
        points=selected
    for i,(x,y) in enumerate(points):
        col=(75,235,75) if i==0 else (245,180,90)
        center=(round(x*zoom),round(y*zoom))
        cv2.circle(drawing,center,8,col,2,cv2.LINE_AA)
        cv2.putText(drawing,'Hip surface' if i==0 else 'Knee top',
                    (max(5,center[0]-55),max(20,center[1]-14)),cv2.FONT_HERSHEY_SIMPLEX,.48,col,1,cv2.LINE_AA)
    if len(points)==2:
        (hx,hy),(_,ky)=points
        # Thin knee reference and offset vertical depth measure.
        yy=round(ky*zoom)
        cv2.line(drawing,(0,yy),(drawing.shape[1]-1,yy),(245,180,90),1,cv2.LINE_AA)
        xx=max(12,min(drawing.shape[1]-12,round(hx*zoom)+35))
        cv2.line(drawing,(xx,round(hy*zoom)),(xx,yy),(70,255,160),2,cv2.LINE_AA)
        diff=hy-ky
        cv2.putText(drawing,f'Signed margin {diff:+.1f} px',
                    (8,drawing.shape[0]-14),cv2.FONT_HERSHEY_SIMPLEX,.47,(90,255,140),1,cv2.LINE_AA)
    panel=drawing.copy()
    cv2.rectangle(panel,(0,0),(drawing.shape[1],72),(8,14,22),-1)
    cv2.addWeighted(panel,.65,drawing,.35,0,drawing)
    cv2.putText(drawing,f'Frame {frame_id} | {frame_id/fps:.3f}s | A/D frame',
                (8,24),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
    hint = ('Click KNEE TOP (point 2/2)' if len(selected)==1
            else 'Click HIP then KNEE | S save | Esc cancel')
    cv2.putText(drawing,hint,
                (8,48),cv2.FONT_HERSHEY_SIMPLEX,.37,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(drawing,'R/right-click clear | +/- zoom',
                (8,66),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1,cv2.LINE_AA)
    return drawing


def save_outputs(video: Path, output: Path, fps: float, w: int, h: int,
                 labels: dict[int,ManualPoint]):
    save_manual(output,video.name,w,h,fps,list(labels.values()))
    artifact_dir=output.parent/(output.stem+'_review')
    artifact_dir.mkdir(parents=True,exist_ok=True)
    shot_dir=artifact_dir/'screenshots'
    shot_dir.mkdir(parents=True,exist_ok=True)
    cap=open_video(str(video))
    rows=[]
    try:
        for frame_id,label in sorted(labels.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
            ok,img=cap.read()
            if not ok:raise ValueError(f'Cannot read source video frame {frame_id}')
            metrics=measure(apply_manual(FrameMetrics(frame_id,frame_id/fps),label,w,h),w,h)
            shot=render_manual(img,label,[],frame_id,fps,1.0)
            filename=f'manual_{frame_id:06d}_{frame_id/fps:.3f}s.png'
            if not cv2.imwrite(str(shot_dir/filename),shot):raise ValueError('Failed to save screenshot')
            row=metrics.to_dict();row['screenshot']=str(Path(shot_dir.name)/filename)
            rows.append(row)
    finally:cap.release()
    if rows:
        path=artifact_dir/'manual_depth_results.csv'
        with path.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    (artifact_dir/'manual_depth_results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    measurements=[measure(apply_manual(FrameMetrics(p.frame,p.frame/fps),p,w,h),w,h)
                  for p in sorted(labels.values(),key=lambda q:q.frame)]
    plot_depth(measurements,artifact_dir/'manual_depth_vs_time.svg')
    return rows


def collect_gui(video: Path, fps: float, width: int, height: int,
                labels: dict[int,ManualPoint], start: float, end: float,
                default_uncertainty: float, proposals: dict | None = None,
                initial_frame: int | None = None, rep_number: int | None = None) -> dict[int,ManualPoint] | None:
    """Interactive frame editor. Return a *copy* of labels only after explicit Save.

    Mouse callbacks run during waitKeyEx. An indefinite wait does not redraw on
    clicks, so the GUI polls every 25 ms and repaints when an input changes.
    Closing the native window or pressing Esc cancels without rewriting files.
    """
    cap = open_video(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open input video: {video}')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    begin = max(0, round(start * fps))
    finish = min(frame_count - 1, round(end * fps))
    if finish < begin:
        cap.release()
        raise ValueError('No frames in review interval')

    frame_id = min(finish, max(begin, initial_frame if initial_frame is not None else begin))
    proposals = proposals or {}
    dismissed = set()
    # WINDOW_AUTOSIZE keeps callback pixel coordinates consistent with the
    # image returned by render_manual. A resizeable WINDOW_NORMAL introduces
    # another GUI scaling factor, especially on Windows high-DPI displays.
    zoom = min(1.0, 620.0 / height, 860.0 / width)
    zoom = max(0.5, zoom)
    selections: list[tuple[float,float]] = []
    working_labels = dict(labels)  # don't mutate annotations unless saved
    window = 'SquatVision: manual IPF depth | S=SAVE | Esc=Cancel'
    dirty = True
    loaded_frame: int | None = None
    image: np.ndarray | None = None

    def click(evt: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal selections, dirty
        if evt == cv2.EVENT_LBUTTONDOWN:
            # Ignore clicks outside the rendered image, rather than silently
            # clamping to the edge and producing invalid anatomical labels.
            if not (0 <= x < round(width*zoom) and 0 <= y < round(height*zoom)):
                return
            xy = (min(width - 1.0, x/zoom), min(height - 1.0, y/zoom))
            if frame_id in working_labels:
                working_labels.pop(frame_id)
                selections = []
            selections.append(xy)
            if len(selections) == 2:
                working_labels[frame_id] = ManualPoint(
                    frame_id, selections[0], selections[1], default_uncertainty, note=f"rep_number={rep_number}; manually positioned" if rep_number else "manually positioned")
                selections = []
            dirty = True
        elif evt == cv2.EVENT_RBUTTONDOWN:
            working_labels.pop(frame_id, None)
            dismissed.add(frame_id)
            selections = []
            dirty = True

    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, click)
        while True:
            if dirty:
                if loaded_frame != frame_id:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ok, image = cap.read()
                    if not ok:
                        raise RuntimeError(f'Cannot read video frame {frame_id}')
                    loaded_frame = frame_id
                display = render_manual(
                    image, working_labels.get(frame_id), selections, frame_id, fps, zoom)
                proposal = proposals.get(frame_id)
                if proposal and frame_id not in working_labels and frame_id not in dismissed and not selections:
                    for part in ('hip','knee'):
                        xy=proposal.get(part,{}).get('point')
                        if xy is not None:
                            cv2.drawMarker(display,tuple(round(v*zoom) for v in xy),
                                           (0,220,255),cv2.MARKER_DIAMOND,14,2)
                    cv2.putText(display,'UNVERIFIED proposal | U accept pair | click to replace',
                                (8,90),cv2.FONT_HERSHEY_SIMPLEX,.34,(0,220,255),1,cv2.LINE_AA)
                cv2.imshow(window, display)
                dirty = False

            # Timed event processing lets the callback trigger an immediate redraw
            # and makes native window-close events observable.
            key = cv2.waitKeyEx(25)
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return None  # Window X: cancel. Never call imshow again.
            except cv2.error:
                return None  # The GUI destroyed the window while waiting.

            if key == 27:  # Escape: cancel without saving or changing existing JSON.
                return None
            if key in (ord('s'), ord('S'), 13, 10, 32):  # Save: S/Enter/Space.
                if not working_labels:
                    print('No completed annotations to save. Click HIP then KNEE, or press Esc to cancel.')
                    continue
                if selections:
                    print('Complete the second point or right-click to clear before saving.')
                    continue
                return working_labels

            if key in (ord('u'), ord('U')) and not selections and frame_id not in working_labels and frame_id not in dismissed:
                proposal = proposals.get(frame_id)
                if proposal:
                    accepted = accept_proposal(proposal, default_uncertainty, width, height, rep_number)
                    if accepted is not None:
                        working_labels[frame_id] = accepted
                        dirty = True
                continue
            next_frame = frame_id
            if key in (ord('a'), ord('A'), 2424832, 65361):
                next_frame = max(begin, frame_id - 1)
            elif key in (ord('d'), ord('D'), 2555904, 65363):
                next_frame = min(finish, frame_id + 1)
            elif key in (ord('j'), ord('J')):
                next_frame = max(begin, frame_id - 10)
            elif key in (ord('l'), ord('L')):
                next_frame = min(finish, frame_id + 10)
            elif key in (ord('r'), ord('R')):
                working_labels.pop(frame_id, None)
                dismissed.add(frame_id)
                selections = []
                dirty = True
            elif key in (ord('+'), ord('=')):
                zoom = min(3.0, zoom + 0.2)
                dirty = True
            elif key in (ord('-'), ord('_')):
                zoom = max(0.5, zoom - 0.2)
                dirty = True
            if next_frame != frame_id:
                frame_id = next_frame
                selections = []
                dirty = True
    finally:
        cap.release()
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass  # Already closed with the title-bar X.


def accept_proposal(proposal, uncertainty, width, height, rep_number=None):
    """Explicit human acceptance only; partial pairs cannot become labels."""
    from squatvision.ipf_depth import validate_manual_point
    hip=proposal.get('hip',{}).get('point')
    knee=proposal.get('knee',{}).get('point')
    if hip is None or knee is None:
        return None
    point=ManualPoint(int(proposal['frame']),tuple(hip),tuple(knee),uncertainty,
                      note=f"rep_number={rep_number}; explicitly accepted automatic proposal")
    validate_manual_point(point,width,height)
    return point


def main():
    p=argparse.ArgumentParser(description='Manually annotate IPF surface references at video frames')
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('manual_depth.json'))
    p.add_argument('--start-time',type=float,default=None,
                   help='Review start; defaults to 0 or auto-selected reference window')
    p.add_argument('--end-time',type=float,default=None,
                   help='Review end; defaults to video end or reference window')
    p.add_argument('--analysis-summary',type=Path,default=None,
                   help='summary.json from app.py; review its detected minimum knee-angle reference')
    p.add_argument('--open-best-candidate',action='store_true',
                   help='Open at the nearest informative raw candidate; still requires manual verification')
    p.add_argument('--rep-number',type=int,default=1,
                   help='Choose a repetition from --analysis-summary (default: 1)')
    p.add_argument('--uncertainty-px',type=float,default=6.)
    p.add_argument('--point',action='append',help='Headless/additional annotation FRAME,HIP_X,HIP_Y,KNEE_X,KNEE_Y')
    p.add_argument('--no-gui',action='store_true',help='Only write provided --point annotations')
    a=p.parse_args()
    cap=open_video(str(a.input))
    if not cap.isOpened():p.error('Cannot open input video')
    fps=cap.get(cv2.CAP_PROP_FPS);w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    if fps<=0 or count<1:p.error('Cannot determine video duration')
    duration=(count-1)/fps
    selected_time=None
    proposals={}
    initial_frame=None
    if a.analysis_summary is not None:
        report=json.loads(a.analysis_summary.read_text(encoding='utf-8'))
        if report.get('resolution') != [w,h] or abs(report.get('fps',0)-fps) > .05:
            p.error('Analysis summary resolution/FPS differ from source video')
        if report.get('source_sha256'):
            import hashlib
            with a.input.open('rb') as source:
                if hashlib.file_digest(source,'sha256').hexdigest() != report['source_sha256']:
                    p.error('Analysis summary fingerprint differs from source video')
        if report.get('source')!=a.input.name:
            p.error('Analysis summary is for a different source video')
        refs=report.get('auto_bottom_reference_frames',[])
        ref=next((r for r in refs if r.get('rep_number')==a.rep_number),None)
        if ref is None or ref.get('timestamp_s') is None:
            p.error('No valid minimum-knee-angle reference for selected repetition')
        selected_time=float(ref['timestamp_s'])
        initial_frame=int(ref['frame'])
        if a.open_best_candidate and ref.get('suggested_review_frame') is not None:
            initial_frame=int(ref['suggested_review_frame'])
        proposals={int(r['frame']):r for r in ref.get('candidates', [])}
    start=max(0., selected_time-1.) if selected_time is not None else 0.
    end=min(duration, selected_time+1.) if selected_time is not None else duration
    start=a.start_time if a.start_time is not None else start
    end=a.end_time if a.end_time is not None else end
    if not (0<=start<=duration and 0<=end<=duration and start<end):
        p.error('Review interval outside source video or has non-positive length')
    if not 0<a.uncertainty_px<max(w,h):p.error('Invalid --uncertainty-px')
    labels=load_manual(a.output,a.input.name,w,h,fps) if a.output.exists() else {}
    for value in a.point or []:
        fields=value.split(',')
        if len(fields)!=5:p.error('--point must contain 5 comma-separated numbers')
        try:
            frame_id=int(fields[0]);hx,hy,kx,ky=map(float,fields[1:])
        except ValueError:p.error('Invalid --point values')
        labels[frame_id]=ManualPoint(frame_id,(hx,hy),(kx,ky),a.uncertainty_px)
    if not a.no_gui:
        updated=collect_gui(a.input,fps,w,h,labels,start,end,a.uncertainty_px,
                            proposals,initial_frame,a.rep_number if a.analysis_summary else None)
        if updated is None:
            print('Manual annotation cancelled. Existing files were not changed.')
            return
        labels=updated
    elif not labels:
        p.error('--no-gui requires at least one --point or existing annotations')
    rows=save_outputs(a.input,a.output,fps,w,h,labels)
    print(f'Saved {len(rows)} annotated frames to {a.output}')
    for row in rows:
        print(f"{row['timestamp_s']:.3f}s: margin={row['depth_margin_px']:+.1f}px "+
              f"status={row['depth_status']} source={row['measurement_source']}")

if __name__=='__main__':main()
