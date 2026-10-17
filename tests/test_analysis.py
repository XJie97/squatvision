import random
import unittest

from squatvision.analysis import FrameMetrics, Config, analyse
from squatvision.geometry import angle_degrees
from squatvision.render import screenshot_name, draw_angle_guides
from squatvision.analysis import PhaseEvent
import cv2
import numpy as np

FPS=30


def create_video_sequence(segments, rng_seed=4):
    """segments: tuples (angle, hip_y, n_frames) or (start_angle,start_y,end_angle,end_y,n)."""
    r=random.Random(rng_seed)
    frames=[]
    for item in segments:
        if len(item)==3:
            k,h,n=item
            entries=[(k,h) for _ in range(n)]
        else:
            a,ha,b,hb,n=item
            entries=[(a+(b-a)*i/(n-1),ha+(hb-ha)*i/(n-1)) for i in range(n)]
        for knee,hip in entries:
            i=len(frames)
            k=knee+r.uniform(-1.4,1.4)
            h=hip+r.uniform(-.002,.002)
            frames.append(FrameMetrics(i,i/FPS,knee_deg=k,hip_deg=k-8,
                                       hip_y=h,knee_y=.61,visibility=.97,side='right'))
    return frames


def base_squat(pause_frames=18):
    return [
        (167,.31,65),
        (167,.31,92,.555,32),
        (92,.555,pause_frames),
        (92,.555,167,.31,42),
        (167,.31,50),
    ]


class TestGeometry(unittest.TestCase):
    def test_angles(self):
        self.assertAlmostEqual(angle_degrees((-1,0),(0,0),(1,0)),180.)
        self.assertAlmostEqual(angle_degrees((1,0),(0,0),(0,1)),90.)
        self.assertIsNone(angle_degrees((0,0),(0,0),(1,0)))


class TestAnalysis(unittest.TestCase):
    def test_one_rep_not_seven_with_knee_jitter(self):
        parts=[(167,.31,120),
               # harmless shallow knee bends prior to actual squat
               (167,.31,136,.335,18),(136,.335,167,.31,18),
               (167,.31,140,.33,15),(140,.33,167,.31,15),
               (167,.31,70)] + base_squat(16) + [(167,.31,90)]
        analysis=analyse(create_video_sequence(parts))
        self.assertEqual(len(analysis.completed),1)
        self.assertEqual(len(analysis.attempts),1)
        self.assertEqual(analysis.attempts[0].outcome,'complete')

    def test_temporal_phases_order_and_pause(self):
        a=analyse(create_video_sequence(base_squat(30)))
        self.assertEqual(len(a.completed),1)
        rep=a.completed[0]
        events={e.name:e for e in rep.events}
        self.assertEqual(list(events),['descent_start','descent_end','ascent_start','lockout'])
        self.assertLess(events['descent_start'].time_s,events['descent_end'].time_s)
        self.assertLess(events['descent_end'].time_s,events['ascent_start'].time_s)
        self.assertLess(events['ascent_start'].time_s,events['lockout'].time_s)
        self.assertGreater(rep.bottom_pause_s,.5)

    def test_no_pause(self):
        a=analyse(create_video_sequence(base_squat(1)))
        self.assertEqual(len(a.completed),1)
        self.assertEqual(a.completed[0].bottom_pause_s,0.)

    def test_two_repetitions(self):
        a=analyse(create_video_sequence(base_squat(12)+base_squat(12)))
        self.assertEqual(len(a.completed),2)
        self.assertEqual([rep.number for rep in a.attempts],[1,2])

    def test_shallow_is_not_rep(self):
        a=analyse(create_video_sequence([(165,.30,60),(165,.30,135,.34,28),
                                         (135,.34,165,.30,30),(165,.30,60)]))
        self.assertEqual(len(a.completed),0)

    def test_deep_crouch_without_lockout_not_completed(self):
        a=analyse(create_video_sequence([(165,.3,50),(165,.3,85,.57,25),(85,.57,35)]))
        self.assertEqual(len(a.completed),0)
        self.assertEqual(len(a.attempts),1)
        self.assertEqual(a.attempts[0].outcome,'incomplete')
        self.assertFalse(any(e.name=='lockout' for e in a.attempts[0].events))

    def test_suspected_failure_when_attempt_reverses(self):
        a=analyse(create_video_sequence([(165,.3,70),(165,.3,89,.56,30),(89,.56,12),
                                         (89,.56,126,.46,16),
                                         (126,.46,93,.55,20),(93,.55,35)]))
        self.assertEqual(len(a.completed),0)
        self.assertEqual(len(a.attempts),1)
        self.assertEqual(a.attempts[0].outcome,'suspected_failure')
        self.assertIn('suspected_failure',[e.name for e in a.attempts[0].events])

    def test_long_tracking_loss_not_a_rep(self):
        frames=create_video_sequence(base_squat(12))
        frames=[FrameMetrics(f.frame,f.time_s) if 85<=f.frame<=150 else f for f in frames]
        a=analyse(frames)
        self.assertEqual(len(a.completed),0)

    def test_stable_upright_prevents_double_counting_bounce(self):
        a=analyse(create_video_sequence([(165,.31,70),
                                         (165,.31,98,.55,28),
                                         (98,.55,145,.52,14),
                                         (145,.52,88,.56,12),
                                         (88,.56,165,.31,35),
                                         (165,.31,50)]))
        self.assertEqual(len(a.completed),1)

    def test_incomplete_when_video_cropped(self):
        frames=create_video_sequence([(165,.31,50),(165,.31,90,.55,29),
                                      (90,.55,15),(90,.55,131,.43,15)])
        a=analyse(frames)
        self.assertEqual(len(a.completed),0)
        self.assertEqual(len(a.attempts),1)


class TestRendering(unittest.TestCase):
    def test_dash_guides_preserve_joint_pixel_and_draw_segments(self):
        img=np.full((640,352,3),120,dtype=np.uint8)
        frame=FrameMetrics(10,.333,knee_deg=95,hip_deg=90,hip_y=.5,knee_y=.66,
                           points=((.45,.26),(.45,.48),(.55,.67),(.6,.89)))
        center=(int(round(.45*352)),int(round(.48*640)))
        original=img[center[1],center[0]].copy()
        draw_angle_guides(img,frame)
        # Knee/hip centres are only hollow rings, no opaque pixel covering.
        self.assertTrue(np.array_equal(img[center[1],center[0]],original))
        self.assertGreater(np.count_nonzero(img!=120),300)

    def test_screenshot_filename_contains_phase_and_time(self):
        value=screenshot_name(2,PhaseEvent('descent_end',12.345,370))
        self.assertIn('attempt_02_descent_end_',value)
        self.assertTrue(value.endswith('.png'))


if __name__=='__main__':
    unittest.main()
