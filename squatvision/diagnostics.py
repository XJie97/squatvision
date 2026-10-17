"""Dependency-free diagnostic SVGs. Missing samples are not connected across gaps."""
from __future__ import annotations
from html import escape
from pathlib import Path
from .analysis import FrameMetrics

COLORS=['#2467a1','#da7437','#448e75','#9255aa']


def _svg_plot(path: Path, title: str, rows: list[FrameMetrics], curves: list[tuple[str,list[float|None]]]):
    if not rows:
        return
    width,height=1000,390
    pad_l,pad_r,pad_t,pad_b=74,30,52,74
    ts=[f.time_s for f in rows]
    t0,t1=min(ts),max(ts)
    if t0==t1:
        t1+=1
    values=[v for _,xs in curves for v in xs if v is not None]
    if not values:
        values=[0,1]
    low,high=min(values),max(values)
    spread=max(high-low,1e-3)
    low-=.08*spread; high+=.08*spread
    def xx(t):return pad_l+(t-t0)/(t1-t0)*(width-pad_l-pad_r)
    def yy(v):return height-pad_b-(v-low)/(high-low)*(height-pad_t-pad_b)
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{pad_l}" y="27" font-family="sans-serif" font-size="19" fill="#263643">{escape(title)}</text>',
           f'<path d="M{pad_l} {pad_t}V{height-pad_b}H{width-pad_r}" stroke="#4d5760" fill="none"/>']
    for i in range(5):
        y=pad_t+i*(height-pad_t-pad_b)/4
        v=high-i*(high-low)/4
        lines.append(f'<line x1="{pad_l}" x2="{width-pad_r}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e6ec"/>')
        lines.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#444">{v:.2f}</text>')
        x=xx(t0+(t1-t0)*i/4)
        lines.append(f'<text x="{x:.1f}" y="{height-pad_b+19}" text-anchor="middle" font-size="12" fill="#444">{t0+(t1-t0)*i/4:.1f}s</text>')
    lines.append(f'<text x="{width/2}" y="{height-22}" text-anchor="middle" font-size="13">Time (seconds)</text>')
    for series,(name,data) in enumerate(curves):
        color=COLORS[series%len(COLORS)]
        chunk=[]
        for t,v in zip(ts,data):
            if v is None:
                if len(chunk)>1:
                    lines.append(f'<polyline points="{" ".join(chunk)}" fill="none" stroke="{color}" stroke-width="2"/>')
                chunk=[]
            else:
                chunk.append(f'{xx(t):.1f},{yy(v):.1f}')
        if len(chunk)>1:
            lines.append(f'<polyline points="{" ".join(chunk)}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_x=pad_l+series*215
        lines.append(f'<line x1="{legend_x}" x2="{legend_x+24}" y1="{height-45}" y2="{height-45}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x+30}" y="{height-40}" font-size="12" fill="#263643">{escape(name)}</text>')
    lines.append('</svg>')
    path.write_text('\n'.join(lines),encoding='utf-8')


def export_plots(frames: list[FrameMetrics], output: Path, h: int, w: int):
    output.mkdir(parents=True,exist_ok=True)
    def x(p,i,size):return None if p is None else p[i]*size
    curves={
        'hip_crease_x.svg':('Hip crease horizontal trajectory (pixels)', [('Hip crease X',[x(f.crease_point,0,w) for f in frames])]),
        'hip_crease_y.svg':('Hip crease vertical trajectory (pixels, downward positive)', [('Hip crease Y',[x(f.crease_point,1,h) for f in frames])]),
        'knee_top_xy.svg':('Knee top trajectories (pixels)', [('Knee top X',[x(f.knee_top_point,0,w) for f in frames]),('Knee top Y',[x(f.knee_top_point,1,h) for f in frames])]),
        'depth_difference.svg':('Hip crease minus knee top (pixels; positive=lower)', [('Vertical difference',[f.crease_to_knee_top_delta*h if f.crease_to_knee_top_delta is not None else None for f in frames])]),
        'tracking_confidence.svg':('Tracking confidence (0..1)', [('Hip crease',[f.hip_crease_tracking_confidence for f in frames]),('Knee top',[f.knee_top_tracking_confidence for f in frames])]),
    }
    for name,(title,data) in curves.items():
        _svg_plot(output/name,title,frames,data)
