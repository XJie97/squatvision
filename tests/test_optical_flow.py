"""Test optical-flow movement against known frame-by-frame synthetic ground truth."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from squatvision.analysis import FrameMetrics
from squatvision.depth import DepthCalibration, classify_depth
from squatvision.optical_flow import LocalPointTracker, track_anatomy
from squatvision.render import draw_depth_guides

H,W=320,240


def make_background():
    rng=np.random.default_rng(95)
    canvas=np.full((H,W),30,np.uint8)
    for cx,cy in [(70,116),(140,174)]:
        roi=rng.integers(75,245,(55,55),dtype=np.uint8)
        canvas[cy-27:cy+28,cx-27:cx+28]=roi
    return cv2.cvtColor(canvas,cv2.COLOR_GRAY2BGR)


def make_video(path, count=20):
    original=make_background()
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),30,(W,H))
    if not writer.isOpened():
        raise RuntimeError('Video encoder unavailable')
    frames=[]
    for i in range(count):
        # Hip point moves positively along both image axes. Knee shifts too,
        # but with a distinct trajectory, so the depth difference changes.
        hdx,hdy=1.0*i,1.6*i
        kdx,kdy=-.5*i,.35*i
        image=np.full((H,W,3),30,np.uint8)
        for xy,dx,dy in [((70,116),hdx,hdy),((140,174),kdx,kdy)]:
            crop=original[xy[1]-27:xy[1]+28,xy[0]-27:xy[0]+28]
            start=(round(xy[0]+dx)-27,round(xy[1]+dy)-27)
            image[start[1]:start[1]+55,start[0]:start[0]+55]=crop
        writer.write(image)
        pts=((.28,.10),((70+hdx)/W,(116+hdy)/H),
             ((140+kdx)/W,(174+kdy)/H),(.85,.90))
        frames.append(FrameMetrics(i,i/30,knee_deg=100,hip_deg=100,side='right',
                                   points=pts,hip_y=pts[1][1],knee_y=pts[2][1]))
    writer.release()
    return frames


class TestOpticalFlow(unittest.TestCase):
    def test_tracks_both_xy_axes_and_relative_height(self):
        with tempfile.TemporaryDirectory() as directory:
            video=Path(directory)/'texture.mp4'
            frames=make_video(video)
            ref=frames[10]
            cal=DepthCalibration.from_reference(ref,(80.,132.),(135.,178.),W,H)
            result=track_anatomy(video,frames,cal,W,H)
            for idx in [0,4,10,15,19]:
                self.assertIsNotNone(result[idx].crease_point,idx)
                self.assertIsNotNone(result[idx].knee_top_point,idx)
            first,last=result[0],result[-1]
            self.assertGreater((last.crease_point[0]-first.crease_point[0])*W,12)
            self.assertGreater((last.crease_point[1]-first.crease_point[1])*H,20)
            self.assertLess((last.knee_top_point[0]-first.knee_top_point[0])*W,-5)
            self.assertGreater((last.knee_top_point[1]-first.knee_top_point[1])*H,3)
            self.assertGreater(last.crease_to_knee_top_delta,first.crease_to_knee_top_delta)
            image=np.full((H,W,3),90,np.uint8)
            draw_depth_guides(image,last)
            self.assertGreater(np.count_nonzero(image!=90),30)

    def test_no_fake_measurement_when_no_texture(self):
        flat=np.full((H,W),128,np.uint8)
        tr=LocalPointTracker(flat,(100,120))
        self.assertFalse(tr.active)
        self.assertEqual(tr.confidence,0)

    def test_depth_status_and_uncertainty(self):
        self.assertEqual(classify_depth(15,100,.9),'below_estimate')
        self.assertEqual(classify_depth(-15,100,.9),'above_estimate')
        self.assertEqual(classify_depth(3,100,.9),'near_threshold_uncertain')
        self.assertEqual(classify_depth(15,100,.25),'uncertain')
        self.assertEqual(classify_depth(None,100,.9),'unavailable')

    def test_additional_anchor_restores_tracking_after_occlusion(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'raw.mp4'
            original=make_video(raw,24)
            capture=cv2.VideoCapture(str(raw))
            altered=Path(d)/'occluded.mp4'
            writer=cv2.VideoWriter(str(altered),cv2.VideoWriter_fourcc(*'mp4v'),30,(W,H))
            for i in range(24):
                ok,image=capture.read()
                self.assertTrue(ok)
                if 9<=i<=12:
                    image[:]=30
                writer.write(image)
            capture.release();writer.release()
            reference=original[0]
            cal=DepthCalibration.from_reference(reference,(70,116),(140,174),W,H)
            without=track_anatomy(altered,original,cal,W,H)
            self.assertIsNone(without[20].crease_to_knee_top_delta)
            anchored=replace(cal,additional_references=((15,85,140,132.5,179.25),))
            with_keyframe=track_anatomy(altered,original,anchored,W,H)
            self.assertIsNotNone(with_keyframe[20].crease_to_knee_top_delta)
            self.assertIsNone(with_keyframe[11].crease_to_knee_top_delta)
            anchored.validate()
            path=Path(d)/'extra.json'
            anchored.save(path)
            self.assertEqual(DepthCalibration.load(path),anchored)

    def test_wrong_subject_side_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            video=Path(d)/'texture.mp4'
            frames=make_video(video)
            cal=DepthCalibration.from_reference(frames[10],(80,132),(135,178),W,H)
            bad=[FrameMetrics(f.frame,f.time_s,side='left',points=f.points,knee_deg=f.knee_deg)
                 for f in frames]
            with self.assertRaisesRegex(ValueError,'No valid corresponding body pose'):
                track_anatomy(video,bad,cal,W,H)


if __name__=='__main__':
    unittest.main()
