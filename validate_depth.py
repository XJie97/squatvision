"""Compare output trajectories with manually labelled source-video frames.

Labels CSV columns (absolute SOURCE pixels):
frame,hip_crease_x,hip_crease_y,knee_top_x,knee_top_y

Reports per-point Euclidean error, signed-depth error, coverage and mean,
median, p90 errors. Unavailable points count against coverage, not error.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def read(path):
    with path.open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def measure(predicted: list[dict], truth: list[dict]):
    by_frame={int(r['frame']):r for r in predicted}
    observed=[]
    missing=0
    for row in truth:
        i=int(row['frame'])
        if i not in by_frame:
            missing+=1;continue
        estimate=by_frame[i]
        try:
            errors={}
            for prefix in ('hip_crease','knee_top'):
                # v4 exports hip_crease_x and hip_crease_y in pixels,
                # knee_top_x_px and knee_top_y_px; legacy x/y are normalised.
                xkey=prefix+'_x' if prefix=='hip_crease' else 'knee_top_x_px'
                ykey=prefix+'_y' if prefix=='hip_crease' else 'knee_top_y_px'
                import math
                dx=float(estimate[xkey])-float(row[prefix+'_x'])
                dy=float(estimate[ykey])-float(row[prefix+'_y'])
                errors[prefix+'_error_px']=round(math.hypot(dx,dy),3)
            true_depth=float(row['hip_crease_y'])-float(row['knee_top_y'])
            inferred=float(estimate['depth_difference_px'])
            errors['depth_error_px']=round(abs(inferred-true_depth),3)
            errors['frame']=i
            observed.append(errors)
        except (KeyError,ValueError,TypeError):
            missing+=1
    report={'labelled_frames':len(truth),'measured_frames':len(observed),
            'unavailable_frames':missing,
            'measurement_coverage':len(observed)/len(truth) if truth else 0.}
    for name in ('hip_crease_error_px','knee_top_error_px','depth_error_px'):
        values=sorted(float(r[name]) for r in observed)
        report[name]={'mean':round(mean(values),3),'median':round(median(values),3),
                      'p90':values[min(len(values)-1,int(.9*(len(values)-1)))]} if values else None
    return report,observed


def main():
    parser=argparse.ArgumentParser(description='Evaluate measured anatomical point trajectories against manual labels')
    parser.add_argument('--frames',type=Path,required=True)
    parser.add_argument('--ground-truth',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/depth_validation.json'))
    args=parser.parse_args()
    report,observed=measure(read(args.frames),read(args.ground_truth))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    detail=args.output.with_suffix('.csv')
    with detail.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['frame','hip_crease_error_px','knee_top_error_px','depth_error_px'])
        writer.writeheader();writer.writerows(observed)
    print(json.dumps(report,indent=2))
    print(f'Details: {detail}')


if __name__=='__main__':main()
