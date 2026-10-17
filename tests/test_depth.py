"""Synthetic calibration, tracking and screenshot-annotation regression tests."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from squatvision.analysis import FrameMetrics
from squatvision.depth import DepthCalibration, apply_depth_calibration, depth_status
from squatvision.render import draw_depth_guides

W, H = 400, 600


def frame(hip_y=.44, knee_y=.66, side='right', valid=True):
    pts=((.30,.19),(.40,hip_y),(.55,knee_y),(.61,.87)) if valid else None
    return FrameMetrics(80,80/30,knee_deg=95 if valid else None,hip_deg=95 if valid else None,
                        hip_y=hip_y if valid else None,knee_y=knee_y if valid else None,
                        side=side,points=pts)


class TestDepthCalibration(unittest.TestCase):
    def test_clicked_points_reconstruct_reference_exactly(self):
        f=frame(hip_y=.54,knee_y=.62)
        crease=(.41*W,.57*H)
        top=(.56*W,.585*H)
        cal=DepthCalibration.from_reference(f,crease,top,W,H)
        result=apply_depth_calibration(f,cal,W,H)
        self.assertAlmostEqual(result.crease_point[0],.41,places=5)
        self.assertAlmostEqual(result.crease_point[1],.57,places=5)
        self.assertAlmostEqual(result.knee_top_point[0],.56,places=5)
        self.assertAlmostEqual(result.knee_top_point[1],.585,places=5)
        self.assertEqual(depth_status(result.crease_to_knee_top_delta),'crease_above_knee_top')

    def test_sign_changes_during_downward_hip_motion(self):
        bottom=frame(.59,.62)
        top=frame(.32,.62)
        cal=DepthCalibration.from_reference(bottom,(.41*W,.615*H),(.56*W,.58*H),W,H)
        low=apply_depth_calibration(bottom,cal,W,H)
        high=apply_depth_calibration(top,cal,W,H)
        self.assertGreater(low.crease_to_knee_top_delta,0)
        self.assertLess(high.crease_to_knee_top_delta,0)
        self.assertEqual(depth_status(low.crease_to_knee_top_delta),'crease_below_knee_top')

    def test_calibration_json_roundtrip(self):
        f=frame(.59,.62)
        cal=DepthCalibration.from_reference(f,(.41*W,.615*H),(.56*W,.58*H),W,H)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'calibration.json'
            cal.save(path)
            self.assertEqual(DepthCalibration.load(path),cal)
            data=json.loads(path.read_text())
            data['side']='nonsense'
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                DepthCalibration.load(path)

    def test_bad_side_and_missing_points_not_interpolated(self):
        f=frame(.59,.62)
        cal=DepthCalibration.from_reference(f,(.41*W,.615*H),(.56*W,.58*H),W,H)
        self.assertIsNone(apply_depth_calibration(frame(side='left'),cal,W,H).crease_point)
        self.assertIsNone(apply_depth_calibration(frame(valid=False),cal,W,H).crease_to_knee_top_delta)
        with self.assertRaises(ValueError):
            apply_depth_calibration(f,cal,W+10,H)

    def test_bad_clicks_rejected(self):
        f=frame()
        with self.assertRaises(ValueError):
            DepthCalibration.from_reference(f,(W*2,0),(.56*W,.6*H),W,H)

    def test_overlay_preserves_reference_centre_pixels(self):
        f=frame(.59,.62)
        cal=DepthCalibration.from_reference(f,(.41*W,.615*H),(.56*W,.58*H),W,H)
        measured=apply_depth_calibration(f,cal,W,H)
        image=np.full((H,W,3),80,dtype=np.uint8)
        crease=(round(measured.crease_point[0]*W),round(measured.crease_point[1]*H))
        knee=(round(measured.knee_top_point[0]*W),round(measured.knee_top_point[1]*H))
        draw_depth_guides(image,measured)
        self.assertTrue(np.array_equal(image[crease[1],crease[0]],[80,80,80]))
        self.assertTrue(np.array_equal(image[knee[1],knee[0]],[80,80,80]))
        self.assertGreater(np.count_nonzero(image!=80),50)


if __name__=='__main__':
    unittest.main()
