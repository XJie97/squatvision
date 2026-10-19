"""Synthetic non-MediaPipe tests for proposed automatic bottom reference labels."""
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cv2

from squatvision.analysis import FrameMetrics
from squatvision.auto_surface import minimum_knee_reference, estimate_surface_candidates


class MinimumKneeTests(unittest.TestCase):
    def test_minimum_is_video_dependent_and_bounded_by_phases(self):
        f = [FrameMetrics(i, float(i), knee_deg=v,
                          points=((.2,.1),(.3,.4),(.7,.5),(.8,.8)))
             for i,v in enumerate([25, 150, 100, 66, 70, 55, 145])]
        events=[SimpleNamespace(name='descent_start',time_s=1.0),
                SimpleNamespace(name='descent_end',time_s=3.),
                SimpleNamespace(name='ascent_start',time_s=4.)]
        a=SimpleNamespace(start_s=1.,end_s=6.,events=events)
        self.assertEqual(minimum_knee_reference(f,a).frame,3)
        # With phases shifted to the second part, the minimum can change.
        a.events=[SimpleNamespace(name='descent_start',time_s=4.),
                  SimpleNamespace(name='ascent_start',time_s=6.)]
        self.assertEqual(minimum_knee_reference(f,a).frame,5)

    def test_missing_knee_data_is_not_fabricated(self):
        a=SimpleNamespace(start_s=1.,end_s=2.,events=[])
        self.assertIsNone(minimum_knee_reference([FrameMetrics(1,1.),FrameMetrics(2,2.)],a))


class SurfaceCandidateTests(unittest.TestCase):
    def setUp(self):
        self.w,self.h=320,240
        self.points=((.2,.2),(.22,.48),(.75,.55),(.8,.8))

    def test_synthetic_thigh_outline_yields_finite_surface_candidates(self):
        hip=np.array([self.points[1][0]*self.w,self.points[1][1]*self.h])
        knee=np.array([self.points[2][0]*self.w,self.points[2][1]*self.h])
        mask=np.zeros((self.h,self.w),np.float32)
        cv2.line(mask,tuple(np.rint(hip).astype(int)),tuple(np.rint(knee).astype(int)),1.,38)
        hp,kp,q,reason=estimate_surface_candidates(mask,self.points,self.w,self.h)
        self.assertIsNotNone(hp,reason)
        self.assertIsNotNone(kp,reason)
        self.assertTrue(.5<=q<=.85)
        self.assertLess(hp[1],hip[1])
        self.assertLess(kp[1],knee[1])

    def test_no_silhouette_boundary_does_not_invent_surface(self):
        mask=np.ones((self.h,self.w),np.float32)
        hp,kp,q,reason=estimate_surface_candidates(mask,self.points,self.w,self.h)
        self.assertIsNone(hp)
        self.assertIsNotNone(reason)

    def test_no_athlete_mask_is_unavailable(self):
        mask=np.zeros((self.h,self.w),np.float32)
        hp,kp,q,reason=estimate_surface_candidates(mask,self.points,self.w,self.h)
        self.assertIsNone(hp)
        self.assertIsNotNone(reason)


class AutomaticFrameAnnotationTests(unittest.TestCase):
    def test_second_pass_evaluates_neighbours_as_unverified(self):
        """Test second-pass integration with realistic MediaPipe *interfaces*.

        Image-boundary extraction and temporal acceptance are tested separately
        in test_surface_evidence.py. Here they are deterministic so this test
        isolates model invocation, multi-frame processing and result export.
        """
        from squatvision.auto_surface import label_knee_minima
        from squatvision.ipf_depth import AUTOMATED
        from squatvision.geometry import JOINTS

        w, h = 320, 240
        pts = ((.2, .2), (.22, .48), (.75, .55), (.8, .8))
        frames = [FrameMetrics(i, float(i)/10, knee_deg=angle, points=pts)
                  for i, angle in enumerate((155., 90., 68., 73., 145.))]
        att = SimpleNamespace(number=1, start_s=0., end_s=.4, events=[
            SimpleNamespace(name='descent_start', time_s=0.),
            SimpleNamespace(name='ascent_start', time_s=.3),
            SimpleNamespace(name='descent_end', time_s=.2)])

        class Capture:
            def isOpened(self): return True
            def set(self, *args): pass
            def read(self): return True, np.zeros((h, w, 3), np.uint8)
            def release(self): pass

        class FakeImage:
            def __init__(self, *, image_format, data):
                self.data = data

        class FakeMask:
            def __init__(self, arr):
                self.data = np.ascontiguousarray(arr, dtype=np.float32)

            def is_contiguous(self):
                return self.data.flags.c_contiguous

            def numpy_view(self):
                return self.data

        # Provide actual coordinates instead of a list containing object().
        pose = [SimpleNamespace(x=.5, y=.5) for _ in range(33)]
        for side_ids in JOINTS.values():
            for joint_id, xy in zip(side_ids, pts):
                pose[joint_id] = SimpleNamespace(x=xy[0], y=xy[1])

        class Result:
            def __init__(self, img):
                self.pose_landmarks = [pose]
                # The mask must match the *padded* image passed to inference.
                self.segmentation_masks = [FakeMask(np.ones(img.data.shape[:2]))]

        class Model:
            def detect(self, img): return Result(img)
            def close(self): pass

        class Task:
            class RunningMode: IMAGE = 1
            PoseLandmarkerOptions = staticmethod(lambda **kw: kw)
            PoseLandmarker = SimpleNamespace(
                create_from_options=lambda opts: Model())

        mp = SimpleNamespace(
            tasks=SimpleNamespace(BaseOptions=lambda **kw: kw, vision=Task),
            Image=FakeImage,
            ImageFormat=SimpleNamespace(SRGB=1))

        hip = np.array(pts[1]) * [w, h]
        knee = np.array(pts[2]) * [w, h]

        def candidate(*args):
            # Surface estimation is a separate unit: return a plausible proposal.
            return {
                'status': 'AUTOMATIC_CANDIDATE',
                'hip': {'point': [float(hip[0]), float(hip[1] - 12)], 'quality': .8},
                'knee': {'point': [float(knee[0]), float(knee[1] - 10)], 'quality': .8},
                'margin_px': float(hip[1] - 12 - (knee[1] - 10)),
                'reason': '',
            }

        def coherent_temporal_review(rows):
            # Independent repeated candidates pass temporal review.
            if sum(r['status'] == 'AUTOMATIC_CANDIDATE' for r in rows) >= 3:
                for r in rows:
                    if r['status'] == 'AUTOMATIC_CANDIDATE':
                        r['status'] = 'AUTOMATIC_UNVERIFIED'
            return rows

        with (patch('squatvision.auto_surface.open_video', return_value=Capture()),
              patch('squatvision.surface_evidence.propose_surfaces', side_effect=candidate) as proposal,
              patch('squatvision.surface_evidence.filter_temporal', side_effect=coherent_temporal_review)):
            new, records = label_knee_minima(
                Path('any.mp4'), frames, SimpleNamespace(attempts=[att]),
                Path('model.task'), mp, w, h)

        self.assertEqual(proposal.call_count, 5)
        self.assertEqual(records[0]['frame'], 2)
        self.assertEqual(records[0]['status'], 'UNVERIFIED_SURFACE_PROPOSAL')
        self.assertIsNotNone(new[2].crease_point)
        self.assertEqual(new[2].measurement_source, AUTOMATED)
        self.assertIsNotNone(new[1].crease_point)
        self.assertEqual(records[0]['evaluated_frames'], 5)
        self.assertEqual(records[0]['candidates'][2]['status'], 'AUTOMATIC_UNVERIFIED')

    def test_noncontiguous_mask_never_calls_numpy_view(self):
        """Do not trigger a fatal MediaPipe float-mask conversion."""
        from squatvision.auto_surface import label_knee_minima

        w, h = 320, 240
        pts = ((.2, .2), (.22, .48), (.75, .55), (.8, .8))
        frames = [FrameMetrics(2, .2, knee_deg=68., points=pts)]
        att = SimpleNamespace(number=1, start_s=.1, end_s=.3, events=[])

        class Capture:
            def isOpened(self): return True
            def set(self, *args): pass
            def read(self): return True, np.zeros((h, w, 3), np.uint8)
            def release(self): pass

        class UnsafeMask:
            def is_contiguous(self): return False
            def numpy_view(self):
                raise AssertionError('Must not read non-contiguous masks')

        class Model:
            def detect(self, *args):
                return SimpleNamespace(segmentation_masks=[UnsafeMask()])
            def close(self): pass

        class Task:
            class RunningMode: IMAGE = 1
            PoseLandmarkerOptions = staticmethod(lambda **kw: kw)
            PoseLandmarker = SimpleNamespace(
                create_from_options=lambda opts: Model())

        mp = SimpleNamespace(
            tasks=SimpleNamespace(BaseOptions=lambda **kw: kw, vision=Task),
            Image=lambda **kw: object(),
            ImageFormat=SimpleNamespace(SRGB=1))

        with patch('squatvision.auto_surface.open_video', return_value=Capture()):
            _, records = label_knee_minima(
                Path('any.mp4'), frames, SimpleNamespace(attempts=[att]),
                Path('model.task'), mp, w, h)

        self.assertEqual(records[0]['status'], 'UNAVAILABLE')
        self.assertEqual(records[0]['candidates'][0]['reason'],
                         'SEGMENTATION_MASK_NONCONTIGUOUS')

    def test_existing_manual_points_are_never_overwritten(self):
        from squatvision.auto_surface import label_knee_minima
        # Tests use ManualPoint source in real repo; here set FrameMetrics directly.
        pts=((.2,.2),(.22,.48),(.75,.55),(.8,.8))
        f=FrameMetrics(4,4.,knee_deg=70.,points=pts,crease_point=(.15,.3),
                       knee_top_point=(.7,.4),measurement_source='MANUAL_ANNOTATION')
        a=SimpleNamespace(number=1,start_s=3.,end_s=5.,events=[])
        class Capture:
            def isOpened(self):return True
            def release(self):pass
        # A segmentation model is initialised but should not be used to replace labels.
        class Model:
            def detect(self,*args):raise AssertionError('do not overwrite manual')
            def close(self):pass
        class Task:
            class RunningMode: IMAGE=1
            PoseLandmarkerOptions=staticmethod(lambda **kw:kw)
            PoseLandmarker=SimpleNamespace(create_from_options=lambda opts:Model())
        mp=SimpleNamespace(tasks=SimpleNamespace(BaseOptions=lambda **kw:kw,vision=Task))
        with patch('squatvision.auto_surface.open_video',return_value=Capture()):
            updated,records=label_knee_minima(Path('any.mp4'),[f],
                SimpleNamespace(attempts=[a]),Path('model.task'),mp,320,240)
        self.assertIs(updated[0],f)
        self.assertEqual(records[0]['status'],'EXISTING_SURFACE_POINTS')


if __name__=='__main__':
    unittest.main()
