"""Regression tests for the falsely confident ABOVE label on partial video coverage."""
import unittest
from dataclasses import replace

import cv2
import numpy as np

from squatvision.analysis import Analysis, FrameMetrics
from squatvision.ipf_depth import (ABOVE, BELOW, UNCERTAIN, UNAVAILABLE, MANUAL, AUTOMATED,
                                    ManualPoint, apply_manual, attempt_depth, measure)
from squatvision.render import overlay, draw_depth_guides

W, H = 352, 640


def sample(i: int, margin: float) -> FrameMetrics:
    f=FrameMetrics(frame=i,time_s=72+i/30,side='right',knee_deg=75.,hip_deg=60.,
                   points=((.30,.42),(.37,.52),(.65,.62),(.77,.82)),
                   crease_point=(.37,(400+margin)/H),knee_top_point=(.65,400/H),
                   depth_tracking_method='local_optical_flow',
                   hip_crease_tracking_confidence=.95,knee_top_tracking_confidence=.92)
    return f


class DepthReportSafetyTests(unittest.TestCase):
    def test_legacy_scoring_remains_available_to_callers(self):
        self.assertEqual(measure(sample(0,-32),W,H).depth_status,ABOVE)

    def test_reporting_never_calls_unverified_flow_above(self):
        m=measure(sample(0,-32),W,H,require_manual_verification=True)
        self.assertEqual(m.depth_margin_px,-32)
        self.assertEqual(m.depth_status,UNCERTAIN)
        self.assertEqual(m.measurement_source,AUTOMATED)
        self.assertIn('not_manually_verified',m.unavailable_reason)

    def test_tracking_loss_at_bottom_invalidates_above_claim(self):
        estimates=[measure(sample(0,-32),W,H,require_manual_verification=True)]
        estimates.extend(measure(FrameMetrics(i,72+i/30),W,H,require_manual_verification=True)
                         for i in range(1,24))
        res=attempt_depth(estimates,71.8,73.5,1,True,bottom_time_s=72.7)['depth_assessment']
        self.assertEqual(res['status'],UNCERTAIN)
        self.assertEqual(res['depth_margin_px'],-32)
        self.assertEqual(res['unavailable_frames'],23)
        self.assertIn('no depth verdict',res['interpretation'].lower())

    def test_manual_above_without_peak_coverage_is_uncertain(self):
        f=FrameMetrics(2,72.067)
        manual=ManualPoint(2,(130,360),(210,400),8.)
        r=attempt_depth([measure(apply_manual(f,manual,W,H),W,H,
                 require_manual_verification=True)],71,74,1,True,bottom_time_s=72.75)['depth_assessment']
        self.assertEqual(r['depth_margin_px'],-40)
        self.assertEqual(r['status'],UNCERTAIN)
        self.assertIn('bottom',r['interpretation'].lower())

    def test_manual_below_remains_valid_observed_measurement(self):
        f=FrameMetrics(12,72.4)
        manual=ManualPoint(12,(130,419),(205,396),8.)
        m=measure(apply_manual(f,manual,W,H),W,H,require_manual_verification=True)
        self.assertEqual(m.measurement_source,MANUAL)
        self.assertEqual(m.depth_status,BELOW)
        self.assertEqual(attempt_depth([m],71,75,1,True,bottom_time_s=72.7)
                         ['depth_assessment']['status'],BELOW)

    def test_unverified_render_does_not_print_est_above(self):
        image=np.full((H,W,3),90,np.uint8)
        frames=Analysis(attempts=[],smoothed_knee=[],smoothed_hip_y=[],
                        upright_knee_deg=None,upright_hip_y=None)
        result=overlay(image,sample(0,-32),frames)
        self.assertEqual(result.shape,image.shape)
        self.assertTrue(np.any(result[55,5:275]!=image[55,5:275]))
        # Text is only in the top band and reference markers; the body below
        # the bracket remains unchanged away from the tracked region.
        self.assertTrue(np.array_equal(result[510,20:320],image[510,20:320]))

    def test_knee_reference_not_across_full_width(self):
        img=np.full((H,W,3),70,np.uint8)
        draw_depth_guides(img,sample(0,5))
        self.assertTrue(np.all(img[400,0:45]==70))
        self.assertTrue(np.any(img[400,140:295]!=70))


if __name__ == '__main__':
    unittest.main()
