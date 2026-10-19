"""Additional v5 tests: IPF surface measurement never uses joint centres as substitutes."""
import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from squatvision.analysis import FrameMetrics
from squatvision.ipf_depth import (ManualPoint, MANUAL, AUTOMATED, ABOVE, BELOW, UNCERTAIN,
                                  UNAVAILABLE, load_manual, save_manual, apply_manual,
                                  measure, attempt_depth)
from squatvision.ipf_plot import plot_depth
from squatvision.render import draw_depth_guides
from manual_depth import save_outputs

W,H=352,640


def pose(i=0,hip_y=0.35,knee_y=.56):
    return FrameMetrics(i,70+i/30,side='right',knee_deg=100.,hip_deg=70.,
                        points=((.35,.12),(.36,hip_y),(.55,knee_y),(.60,.85)),
                        hip_y=hip_y,knee_y=knee_y)


class ManualDepthTests(unittest.TestCase):
    def annotation(self,i=0,hip=(120,340),knee=(220,320),u=6):
        return ManualPoint(i,hip,knee,u)

    def test_below_status_and_separate_surfaces(self):
        # Pose joint centre is far above the manually labelled hip surface.
        f=apply_manual(pose(),self.annotation(),W,H)
        m=measure(f,W,H)
        self.assertAlmostEqual(m.depth_margin_px,20.)
        self.assertEqual(m.depth_status,BELOW)
        self.assertEqual(m.measurement_source,MANUAL)
        self.assertAlmostEqual(m.upper_thigh_at_hip_y,340.)
        self.assertNotAlmostEqual(m.upper_thigh_at_hip_y,f.hip_y*H)
        self.assertAlmostEqual(m.depth_margin_normalized,20 / np.hypot((.36-.55)*W,(.35-.56)*H))

    def test_above_zero_and_uncertainty(self):
        for delta,status in [(-20,ABOVE),(0,UNCERTAIN),(3,UNCERTAIN),(10,BELOW)]:
            a=self.annotation(hip=(90,320+delta),knee=(210,320))
            self.assertEqual(measure(apply_manual(pose(),a,W,H),W,H).depth_status,status)

    def test_same_depth_with_horiz_and_vert_translation(self):
        m1=measure(apply_manual(pose(),self.annotation(0,(100,340),(190,320)),W,H),W,H)
        m2=measure(apply_manual(pose(1),self.annotation(1,(160,380),(220,360)),W,H),W,H)
        self.assertEqual(m1.depth_margin_px,m2.depth_margin_px)
        self.assertNotEqual(m1.upper_thigh_at_hip_x,m2.upper_thigh_at_hip_x)

    def test_relative_motion_changes_depth_independent_of_angles(self):
        p=pose()
        m1=measure(apply_manual(p,self.annotation(0,(100,300),(220,320)),W,H),W,H)
        m2=measure(apply_manual(pose(1),self.annotation(1,(145,344),(205,322)),W,H),W,H)
        self.assertEqual(m1.depth_status,ABOVE)
        self.assertEqual(m2.depth_status,BELOW)
        self.assertEqual(p.knee_deg,pose(1).knee_deg)

    def test_flow_lost_but_manual_still_works(self):
        blank=FrameMetrics(2,70.1,depth_tracking_method='local_optical_flow',
                           depth_unavailable_reason='local_features_missing')
        self.assertEqual(measure(blank,W,H).depth_status,UNAVAILABLE)
        manual=apply_manual(blank,self.annotation(2),W,H)
        self.assertEqual(measure(manual,W,H).depth_status,BELOW)
        self.assertIsNone(measure(manual,W,H).depth_margin_normalized)

    def test_auto_requires_both_points_and_sufficient_confidence(self):
        good=replace(pose(),crease_point=(.4,.55),knee_top_point=(.6,.50),
                     depth_tracking_method='local_optical_flow',
                     hip_crease_tracking_confidence=.92,knee_top_tracking_confidence=.85)
        self.assertEqual(measure(good,W,H).depth_status,BELOW)
        self.assertEqual(measure(replace(good,knee_top_point=None),W,H).depth_status,UNAVAILABLE)
        self.assertEqual(measure(replace(good,hip_crease_tracking_confidence=.21),W,H).depth_status,UNAVAILABLE)

    def test_error_bound_changes_classification(self):
        a=self.annotation(hip=(130,328),knee=(200,320),u=12)
        self.assertEqual(measure(apply_manual(pose(),a,W,H),W,H).depth_status,UNCERTAIN)
        b=replace(a,uncertainty_px=5.)
        self.assertEqual(measure(apply_manual(pose(),b,W,H),W,H).depth_status,BELOW)

    def test_manual_json_roundtrip_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manual.json'
            point=self.annotation()
            save_manual(p,'sample.mp4',W,H,30.,[point])
            got=load_manual(p,'sample.mp4',W,H,30.)
            self.assertEqual(got[0],point)
            with self.assertRaises(ValueError):save_manual(p,'sample.mp4',W,H,30.,[point,point])
            with self.assertRaises(ValueError):load_manual(p,'different.mp4',W,H,30.)
            with self.assertRaises(ValueError):save_manual(p,'sample.mp4',W,H,30.,[self.annotation(2,(-3,3))])

    def test_replacement_of_same_frame_before_saving(self):
        points={5:self.annotation(5)}
        points[5]=self.annotation(5,(130,345),(210,320))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'points.json';save_manual(p,'a.mp4',W,H,30.,list(points.values()))
            self.assertEqual(load_manual(p,'a.mp4',W,H,30.)[5].upper_thigh_at_hip,(130,345))

    def test_maximum_relative_depth_not_minimum_knee_angle(self):
        frames=[pose(i) for i in range(3)]
        margins=[4.,15.,9.]
        obs=[measure(apply_manual(f,self.annotation(f.frame,(140,320+delta),(220,320)),W,H),W,H)
             for f,delta in zip(frames,margins)]
        result=attempt_depth(obs,69.,75.,1,True)['depth_assessment']
        self.assertEqual(result['maximum_depth_frame'],1)
        self.assertEqual(result['depth_margin_px'],15.)
        self.assertEqual(result['valid_frames'],3)
        self.assertTrue(result['interpretation'].startswith('Maximum of observed'))

    def test_no_valid_depth_still_independent_of_rep_outcome(self):
        missing=[measure(pose(i),W,H) for i in range(4)]
        r=attempt_depth(missing,69.,75.,1,True)
        self.assertTrue(r['completed'])
        self.assertEqual(r['depth_assessment']['status'],UNAVAILABLE)

    def test_plot_has_gaps_and_manual_markers(self):
        frames=[pose(i) for i in range(5)]
        frames[0]=replace(frames[0],crease_point=(.4,.6),knee_top_point=(.6,.5),
                          depth_tracking_method='local_optical_flow',
                          hip_crease_tracking_confidence=.9,knee_top_tracking_confidence=.9)
        frames[4]=apply_manual(frames[4],self.annotation(4),W,H)
        ms=[measure(f,W,H) for f in frames]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'depth.svg'
            plot_depth(ms,p)
            svg=p.read_text()
            self.assertIn('Orange: manually annotated',svg)
            self.assertNotIn('<polyline',svg)
            self.assertIn('circle',svg)

    def test_depth_line_rendered(self):
        sample=apply_manual(pose(),self.annotation(),W,H)
        image=np.full((H,W,3),50,np.uint8)
        before=image.copy()
        draw_depth_guides(image,sample)
        self.assertGreater(int(np.abs(image.astype('int16')-before.astype('int16')).sum()),1000)
        self.assertTrue((image[320,:,0]!=50).any())

    def test_headless_real_source_manual_screenshots_and_csv(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            video=root/'source.mp4'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(W,H))
            self.assertTrue(writer.isOpened())
            for i in range(8):writer.write(np.full((H,W,3),80,np.uint8))
            writer.release()
            result=save_outputs(video,root/'manual.json',30,W,H,{3:self.annotation(3)})
            self.assertEqual(len(result),1)
            self.assertEqual(result[0]['measurement_source'],MANUAL)
            self.assertTrue((root/'manual_review'/'manual_depth_results.csv').is_file())
            self.assertEqual(len(list((root/'manual_review'/'screenshots').glob('*.png'))),1)
            with (root/'manual_review'/'manual_depth_results.csv').open() as f:
                self.assertAlmostEqual(float(list(csv.DictReader(f))[0]['depth_margin_px']),20.)

if __name__=='__main__':unittest.main()
