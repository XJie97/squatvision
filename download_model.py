"""Fetch official MediaPipe Lite Pose Landmarker model for first-time setup."""
import argparse
from pathlib import Path
from urllib.request import urlopen

URL='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('models/pose_landmarker_lite.task'))
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        print(f'Already present: {args.output}')
        return
    try:
        with urlopen(URL,timeout=30) as response:
            content=response.read()
        if len(content)<100_000:
            raise RuntimeError('Model download unexpectedly small')
        args.output.write_bytes(content)
        print(f'Downloaded: {args.output}')
    except Exception:
        args.output.unlink(missing_ok=True)
        raise

if __name__=='__main__':
    main()
