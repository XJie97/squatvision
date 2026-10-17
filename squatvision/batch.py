"""Batch processing for independent squat-analysis video recordings.

The batch runner is agnostic to MediaPipe and can be tested without AI models.
Camera-specific calibration/annotation files are never reused across recordings.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable


class BatchInputError(ValueError):
    """Unsafe or invalid batch input configuration."""


def find_mp4_files(input_dir: Path) -> list[Path]:
    """Discover immediate .mp4 files, sorted deterministically by name.

    Check output-folder collisions on case-insensitive Windows filesystems.
    """
    if not input_dir.is_dir():
        raise BatchInputError(f'Input directory does not exist: {input_dir}')
    videos = sorted((p for p in input_dir.iterdir()
                     if p.is_file() and p.suffix.casefold() == '.mp4'),
                    key=lambda p: (p.name.casefold(), p.name))
    if not videos:
        raise BatchInputError(f'No .mp4 videos found directly inside: {input_dir}')
    names = {}
    for video in videos:
        output_stem = video.stem.casefold()
        if output_stem in names:
            raise BatchInputError(
                f'Video filenames would share the same output directory: '
                f'{names[output_stem].name!r} and {video.name!r}. '
                'Rename one of the videos before processing.')
        names[output_stem] = video
    return videos


def _matching_json(folder: Path | None, video: Path) -> Path | None:
    """Only use a file whose stem matches this video exactly (not a generic JSON)."""
    if folder is None:
        return None
    candidate = folder / (video.stem + '.json')
    return candidate if candidate.is_file() else None


def run_batch(input_dir: Path, output_root: Path,
              process_video: Callable[[Path, Path, Path | None, Path | None], None],
              annotations_dir: Path | None = None,
              calibrations_dir: Path | None = None) -> list[dict]:
    """Process all videos, continuing after per-video errors.

    process_video(video, destination, annotation_json, calibration_json) runs
    the existing single-video pipeline. Records successes and failures in
    batch_summary.json/csv. Returns one row for each source video.
    """
    videos = find_mp4_files(input_dir)  # preflight before writing output
    for label, folder in (('Annotation', annotations_dir),
                          ('Calibration', calibrations_dir)):
        if folder is not None and not folder.is_dir():
            raise BatchInputError(f'{label} directory does not exist: {folder}')
    if output_root.resolve() == input_dir.resolve():
        raise BatchInputError('Output root must differ from the input directory.')
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, video in enumerate(videos, start=1):
        destination = output_root / video.stem
        annotation = _matching_json(annotations_dir, video)
        calibration = _matching_json(calibrations_dir, video)
        print(f'[{number}/{len(videos)}] Analysing {video.name} -> {destination}', flush=True)
        row = {
            'video': video.name,
            'status': 'success',
            'output_dir': str(destination),
            'completed_reps': None,
            'total_attempts': None,
            'selected_side': None,
            'usable_pose_fraction': None,
            'depth_statuses': None,
            'manual_annotations_used': bool(annotation),
            'depth_calibration_used': bool(calibration),
            'error': '',
        }
        try:
            process_video(video, destination, annotation, calibration)
            report_path = destination / 'summary.json'
            if not report_path.is_file():
                raise RuntimeError('Analysis finished without writing summary.json')
            report = json.loads(report_path.read_text(encoding='utf-8'))
            row['completed_reps'] = report.get('completed_reps')
            row['total_attempts'] = report.get('total_attempts')
            row['selected_side'] = report.get('selected_side')
            row['usable_pose_fraction'] = report.get('usable_pose_fraction')
            row['depth_statuses'] = ', '.join(
                str(a.get('depth_assessment', {}).get('status', 'UNAVAILABLE'))
                for a in report.get('ipf_depth_assessments', [])
            ) or None
            print(f'  Done: {row["completed_reps"]} completed reps', flush=True)
        except (Exception, SystemExit) as exc:
            # A bad individual video does not abort analysis of the others.
            row['status'] = 'failed'
            row['error'] = str(exc) or type(exc).__name__
            print(f'  Failed: {row["error"]}', flush=True)
        rows.append(row)

    (output_root / 'batch_summary.json').write_text(
        json.dumps(rows, indent=2), encoding='utf-8')
    with (output_root / 'batch_summary.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    succeeded = sum(r['status'] == 'success' for r in rows)
    print(f'Batch complete: {succeeded}/{len(rows)} succeeded. '
          f'Report: {output_root / "batch_summary.csv"}', flush=True)
    return rows
