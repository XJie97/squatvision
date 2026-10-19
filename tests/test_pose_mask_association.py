"""Regression tests for first-pass pose / segmentation association.

These use synthetic masks and a fake MediaPipe IMAGE pass, not real anatomical
accuracy benchmarks. They do not imply an IPF-valid depth determination.
"""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from squatvision.analysis import FrameMetrics
from squatvision.auto_surface import label_knee_minima
from squatvision.geometry import JOINTS
from squatvision.surface_evidence import pose_mask_support


class PoseMaskSupportTests(unittest.TestCase):
    def setUp(self):
        self.w, self.h = 320, 240
        self.pts = ((.2, .2), (.22, .48), (.75, .55), (.8, .8))
        self.hip = np.rint(np.array(self.pts[1]) * [self.w, self.h]).astype(int)
        self.knee = np.rint(np.array(self.pts[2]) * [self.w, self.h]).astype(int)

    def test_contiguous_thigh_is_supported_with_separate_bystander(self):
        mask = np.zeros((self.h, self.w), np.float32)
        cv2.line(mask, tuple(self.hip), tuple(self.knee), 1., thickness=32)
        cv2.rectangle(mask, (0, 30), (25, 215), 1., -1)  # Unconnected bystander.
        result = pose_mask_support(mask, self.pts, self.w, self.h)
        self.assertTrue(result['supported'], result)
        self.assertEqual(result['supported_middle_samples'], 3)

    def test_disconnected_hip_knee_masks_are_rejected(self):
        mask = np.zeros((self.h, self.w), np.float32)
        cv2.circle(mask, tuple(self.hip), 18, 1., -1)
        cv2.circle(mask, tuple(self.knee), 18, 1., -1)
        result = pose_mask_support(mask, self.pts, self.w, self.h)
        self.assertFalse(result['supported'])
        self.assertEqual(result['reason'], 'POSE_MASK_MISMATCH')

    def test_connected_but_missing_thigh_is_rejected(self):
        mask = np.zeros((self.h, self.w), np.float32)
        cv2.circle(mask, tuple(self.hip), 6, 1., -1)
        cv2.circle(mask, tuple(self.knee), 6, 1., -1)
        # A connected path bypasses the thigh's actual centreline.
        path = [tuple(self.hip), (int(self.hip[0]), 15),
                (int(self.knee[0]), 15), tuple(self.knee)]
        cv2.polylines(mask, [np.array(path, np.int32)], False, 1., thickness=3)
        result = pose_mask_support(mask, self.pts, self.w, self.h)
        self.assertFalse(result['supported'])
        self.assertEqual(result['reason'], 'THIGH_CENTERLINE_NOT_IN_MASK')


class CroppedSecondPassTests(unittest.TestCase):
    def _run(self, shift):
        w, h = 320, 240
        pts = ((.2, .2), (.22, .48), (.75, .55), (.8, .8))
        frames = [FrameMetrics(i, i / 10, knee_deg=a, points=pts, side='right')
                  for i, a in enumerate((155., 90., 68., 73., 145.))]
        attempt = SimpleNamespace(number=1, start_s=0., end_s=.4, events=[
            SimpleNamespace(name='descent_start', time_s=0.),
            SimpleNamespace(name='descent_end', time_s=.2),
            SimpleNamespace(name='ascent_start', time_s=.3)])

        class Capture:
            def isOpened(self): return True
            def set(self, *args): pass
            def read(self): return True, np.zeros((h, w, 3), np.uint8)
            def release(self): pass

        class Mask:
            def __init__(self, arr): self.arr = np.ascontiguousarray(arr, dtype=np.float32)
            def is_contiguous(self): return True
            def numpy_view(self): return self.arr

        class Image:
            def __init__(self, image_format, data): self.data = data

        pose = [SimpleNamespace(x=.5, y=.5) for _ in range(33)]
        for idx, xy in zip(JOINTS['right'], pts):
            pose[idx] = SimpleNamespace(x=xy[0] + shift, y=xy[1])

        class Model:
            def detect(self, image):
                return SimpleNamespace(
                    pose_landmarks=[pose],
                    segmentation_masks=[Mask(np.ones(image.data.shape[:2]))])
            def close(self): pass

        class Tasks:
            class RunningMode: IMAGE = 1
            PoseLandmarkerOptions = staticmethod(lambda **kw: kw)
            PoseLandmarker = SimpleNamespace(create_from_options=lambda opts: Model())

        mp = SimpleNamespace(
            tasks=SimpleNamespace(BaseOptions=lambda **kw: kw, vision=Tasks),
            Image=Image, ImageFormat=SimpleNamespace(SRGB=1))

        def fake_candidate(*args):
            return dict(status='AUTOMATIC_CANDIDATE', reason='',
                        hip={'point': [80., 125.], 'quality': .6, 'offset_fraction': .3},
                        knee={'point': [242., 115.], 'quality': .6, 'offset_fraction': .3},
                        margin_px=10.)

        with (patch('squatvision.auto_surface.open_video', return_value=Capture()),
              patch('squatvision.surface_evidence.propose_surfaces', side_effect=fake_candidate)):
            return label_knee_minima(
                Path('mock.mp4'), frames, SimpleNamespace(attempts=[attempt]),
                Path('mock.task'), mp, w, h)

    def test_moderate_crop_pose_disagreement_does_not_block_valid_mask(self):
        updated, records = self._run(0.18)  # ~58 px shift in second pass
        self.assertEqual(records[0]['status'], 'UNVERIFIED_SURFACE_PROPOSAL')
        self.assertIsNotNone(updated[2].crease_point)
        candidate = records[0]['candidates'][2]
        self.assertTrue(candidate['original_pose_mask_support']['supported'])
        self.assertEqual(candidate['second_pass_pose_warning'],
                         'CROP_SENSITIVE_JOINT_DISAGREEMENT')

    def test_extreme_second_pass_person_mismatch_remains_rejected(self):
        updated, records = self._run(0.85)
        self.assertEqual(records[0]['status'], 'UNAVAILABLE')
        self.assertIsNone(updated[2].crease_point)
        self.assertEqual(records[0]['candidates'][2]['reason'],
                         'SECOND_PASS_PERSON_IDENTITY_UNCERTAIN')


if __name__ == '__main__':
    unittest.main()
