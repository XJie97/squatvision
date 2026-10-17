"""Dependency-free diagnostic SVG with genuine gaps and manual markers."""
from __future__ import annotations
from html import escape
from pathlib import Path
from .ipf_depth import Measurement, MANUAL, VERIFIED


def plot_depth(rows: list[Measurement], path: Path) -> None:
    W,H=1100,480
    left,right,top,bottom=80,30,60,80
    valid=[r for r in rows if r.depth_margin_px is not None]
    t0=min((r.timestamp_s for r in rows),default=0.)
    t1=max((r.timestamp_s for r in rows),default=1.)
    if t1<=t0:t1=t0+1
    magnitude=max([15.]+[abs(r.depth_margin_px or 0)+(r.depth_uncertainty_px or 0) for r in valid])*1.13
    def x(t):return left+(t-t0)/(t1-t0)*(W-left-right)
    def y(v):return top+(magnitude-v)/(magnitude*2)*(H-top-bottom)
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<text x="75" y="32" font-size="20" font-family="sans-serif">SquatVision experimental IPF surface depth vs time</text>',
         f'<path d="M{left},{top} V{H-bottom} H{W-right}" fill="none" stroke="#56616b"/>',
         f'<path d="M{left},{y(0):.2f} H{W-right}" stroke="#cc6d2c" stroke-dasharray="6 5"/>',
         f'<text x="{W-right-4}" y="{y(0)-8:.1f}" text-anchor="end" font-size="12" fill="#aa501e">0 px: surfaces level</text>',
         f'<text transform="translate(22,{H/2}) rotate(-90)" text-anchor="middle" font-size="14">Hip surface minus knee top (pixels; positive = lower)</text>',
         f'<text x="{W/2}" y="{H-18}" text-anchor="middle" font-size="14">Video time (seconds)</text>']
    for z in range(5):
        t=t0+(t1-t0)*z/4
        out.append(f'<text x="{x(t):.2f}" y="{H-bottom+24}" font-size="12" text-anchor="middle">{t:.2f}s</text>')
    for v in (-magnitude,0,magnitude):
        out.append(f'<text x="{left-8}" y="{y(v)+5:.2f}" text-anchor="end" font-size="12">{v:+.1f}</text>')
    # Only draw segments between actual consecutive valid AUTO frames. Never
    # connect across missing samples, and never connect manual islands.
    prev=None
    for r in rows:
        if r.depth_margin_px is None or r.measurement_source in (MANUAL,VERIFIED):
            prev=None
            continue
        if prev is not None:
            out.append(f'<line x1="{x(prev.timestamp_s):.2f}" y1="{y(prev.depth_margin_px):.2f}" '
                       f'x2="{x(r.timestamp_s):.2f}" y2="{y(r.depth_margin_px):.2f}" stroke="#397db1" stroke-width="2"/>')
        prev=r
    for r in valid:
        if r.measurement_source not in (MANUAL,VERIFIED):continue
        X=x(r.timestamp_s);Y=y(r.depth_margin_px)
        upper=y(r.depth_margin_px+(r.depth_uncertainty_px or 0))
        lower=y(r.depth_margin_px-(r.depth_uncertainty_px or 0))
        out.extend([f'<line x1="{X:.2f}" y1="{upper:.2f}" x2="{X:.2f}" y2="{lower:.2f}" stroke="#c15c2b" stroke-width="2"/>',
                    f'<circle cx="{X:.2f}" cy="{Y:.2f}" r="5" fill="#df7d35" stroke="white"/>'])
    if valid:
        manual_valid = [r for r in valid if r.measurement_source in (MANUAL, VERIFIED)]
        peak=max(manual_valid or valid,key=lambda r:r.depth_margin_px)
        colour='#2d8751' if manual_valid else '#777777'
        dash='' if manual_valid else ' stroke-dasharray="3 3"'
        out.append(f'<circle cx="{x(peak.timestamp_s):.2f}" cy="{y(peak.depth_margin_px):.2f}" r="7" fill="none" stroke="{colour}" stroke-width="3"{dash}/>')
    else:
        out.append(f'<text x="{W/2}" y="{H/2}" text-anchor="middle" font-size="23" fill="#a44">Insufficient valid anatomical observations</text>')
    out.extend(['<text x="82" y="440" font-size="13" fill="#397db1">Blue: UNVERIFIED optical flow (diagnostic only)</text>',
                '<text x="82" y="460" font-size="13" fill="#c15c2b">Orange: manually annotated observations and engineering error bounds</text>',
                '</svg>'])
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(out),encoding='utf-8')
