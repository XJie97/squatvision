"""Use upright, display-oriented frames consistently in every video consumer."""
import cv2


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    # OpenCV defaults differ by build. Set before querying dimensions or reading.
    rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META) if hasattr(cap, 'get') else 0
    supported = cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1) if hasattr(cap, 'set') else False
    if rotation % 360 and not supported:
        cap.release()
        raise ValueError('Video backend cannot apply rotation metadata; convert to an upright CFR MP4 first')
    return cap
