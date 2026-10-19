"""Synthetic evidence checks are not real anatomical accuracy validation."""
import unittest
from types import SimpleNamespace
from copy import deepcopy
from pathlib import Path
import tempfile
from unittest.mock import patch
import cv2
import numpy as np
from squatvision.surface_evidence import propose_surfaces, filter_temporal, diagnostic_image
from squatvision.auto_surface import bottom_neighbourhood, minimum_knee_reference
from squatvision.analysis import FrameMetrics
from squatvision.ipf_depth import apply_manual, measure, MANUAL, save_manual, load_manual
from squatvision.video import open_video
from manual_depth import accept_proposal

class SurfaceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.w,self.h=320,240
        self.points=((.2,.2),(.25,.5),(.75,.5),(.8,.9))
        self.mask=np.zeros((240,320),np.float32)
        cv2.line(self.mask,(80,120),(240,120),1.,40)
        self.image=np.full((240,320,3),100,np.uint8)
    def detect(self,mask=None,image=None,points=None):
        return propose_surfaces(self.image if image is None else image,
            self.mask if mask is None else mask,self.points if points is None else points,320,240)
    def test_visible_local_contours_at_both_joints(self):
        r=self.detect();self.assertEqual(r['status'],'AUTOMATIC_CANDIDATE')
        self.assertAlmostEqual(r['hip']['point'][0],80,places=2)
        self.assertAlmostEqual(r['knee']['point'][0],240,places=2)
        self.assertLess(abs(r['margin_px']),1)
    def test_torso_overlap_preserves_visible_knee(self):
        self.mask[:140,40:150]=1;r=self.detect()
        self.assertIsNone(r['hip']['point']);self.assertIsNotNone(r['knee']['point'])
        self.assertIsNone(r['margin_px'])
    def test_missing_knee_preserves_hip(self):
        self.mask[:150,200:290]=1;r=self.detect()
        self.assertIsNotNone(r['hip']['point']);self.assertIsNone(r['knee']['point'])
    def test_internal_rgb_boundary_is_only_candidate(self):
        m=np.ones_like(self.mask);im=np.full_like(self.image,180);im[100:]=40
        r=self.detect(m,im);self.assertEqual(r['status'],'AUTOMATIC_CANDIDATE')
        self.assertEqual(r['hip']['method'],'internal_rgb_boundary')
        self.assertAlmostEqual(r['hip']['point'][1],100,delta=3)
    def test_competing_bands_rejected(self):
        im=np.full_like(self.image,180);im[80:100]=40
        r=self.detect(np.ones_like(self.mask),im)
        self.assertIsNone(r['hip']['point']);self.assertIn('AMBIGUOUS',r['hip']['reason'])
    def test_flat_foreground_no_fabrication(self):
        r=self.detect(np.ones_like(self.mask))
        self.assertIsNone(r['hip']['point']);self.assertIsNone(r['knee']['point'])
    def test_partial_occlusion_not_relaxed_sampling(self):
        self.mask[:130,80:86]=1;r=self.detect();self.assertIsNone(r['hip']['point'])
    def test_invalid_masks_and_nan_points(self):
        for m in (np.full_like(self.mask,np.nan),np.full_like(self.mask,2.),np.zeros((10,10))):
            self.assertEqual(self.detect(m)['reason'],'INVALID_SEGMENTATION_MASK')
        p=list(self.points);p[1]=(float('nan'),.5)
        self.assertEqual(self.detect(points=p)['reason'],'INVALID_POSE_JOINT_CENTRES')
    def test_disconnected_and_background_pose_rejected(self):
        self.assertEqual(self.detect(np.zeros_like(self.mask))['reason'],'POSE_MASK_MISMATCH')
        self.mask[:,145:155]=0;self.assertEqual(self.detect()['reason'],'POSE_MASK_MISMATCH')
    def test_short_thigh_rejected(self):
        p=list(self.points);p[2]=(.3,.5)
        self.assertEqual(self.detect(points=p)['reason'],'PROJECTED_THIGH_TOO_SHORT')
    def test_positive_and_negative_margins(self):
        for y,sign in ((100,1),(140,-1)):
            p=list(self.points);p[2]=(.75,y/240)
            m=np.zeros_like(self.mask);cv2.line(m,(80,120),(240,y),1.,40)
            self.assertGreater(sign*self.detect(m,points=p)['margin_px'],0)
    def test_diagnostic_image_preserves_source(self):
        before=self.image.copy();r=self.detect();r.update(frame=3,timestamp_s=.1)
        image=diagnostic_image(self.image,self.mask,self.points,r)
        self.assertGreaterEqual(image.shape[1],1000);np.testing.assert_array_equal(before,self.image)

class TemporalReviewTests(unittest.TestCase):
    def rows(self,fps=30):
        return [dict(frame=i,timestamp_s=i/fps,status='AUTOMATIC_CANDIDATE',margin_px=5.,
                hip={'point':[80,100],'offset_fraction':.12},
                knee={'point':[240,95],'offset_fraction':.12},reason='') for i in range(5)]
    def test_supported_pairs_remain_unverified(self):
        self.assertTrue(all(r['status']=='AUTOMATIC_UNVERIFIED' for r in filter_temporal(self.rows())))
    def test_isolated_pair_rejected_but_visible_for_review(self):
        r=filter_temporal(self.rows()[:1])[0]
        self.assertEqual(r['reason'],'INSUFFICIENT_VALID_FRAMES');self.assertIsNotNone(r['hip']['point'])
    def test_implausible_relative_jump_rejected(self):
        r=self.rows();r[2]['hip']['offset_fraction']=.5
        self.assertEqual(filter_temporal(r)[2]['reason'],'TEMPORAL_INCONSISTENCY')
    def test_large_time_gaps_not_bridged(self):
        self.assertTrue(all(r['status']=='UNAVAILABLE' for r in filter_temporal(self.rows(1))))
    def test_explicit_acceptance_roundtrips_rep_note(self):
        r=self.rows()[0];before=deepcopy(r);p=accept_proposal(r,8,320,240,2)
        self.assertEqual(r,before);self.assertEqual(p.source,MANUAL);self.assertIn('rep_number=2',p.note)
        self.assertEqual(measure(apply_manual(FrameMetrics(0,0.),p,320,240),320,240).measurement_source,MANUAL)
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'labels.json';save_manual(path,'a.mp4',320,240,30,[p])
            self.assertEqual(load_manual(path,'a.mp4',320,240,30)[0],p)
    def test_partial_pair_cannot_be_accepted(self):
        r=self.rows()[0];r['hip']['point']=None
        self.assertIsNone(accept_proposal(r,8,320,240))
    def test_invalid_proposal_coordinates_rejected(self):
        r=self.rows()[0];r['hip']['point']=[900,100]
        with self.assertRaises(ValueError):accept_proposal(r,8,320,240)
    def test_dynamic_multiple_reps_short_video_and_fps(self):
        for fps in (24,30,60):
            fs=[FrameMetrics(i,i/fps,knee_deg=60+abs((i/fps)%3-1)*20,
                points=((.2,.2),(.3,.5),(.7,.5),(.8,.8))) for i in range(6*fps)]
            refs=[]
            for n in range(2):
                a=SimpleNamespace(start_s=n*3,end_s=(n+1)*3-.1,events=[])
                ref=minimum_knee_reference(fs,a);refs.append(ref)
                self.assertTrue(all(a.start_s<=f.time_s<=a.end_s and abs(f.time_s-ref.time_s)<=.5
                                    for f in bottom_neighbourhood(fs,a,ref,.5)))
            self.assertLess(refs[0].time_s,refs[1].time_s)
    def test_invalid_window_rejected(self):
        for v in (0,-1,float('nan')):
            with self.assertRaises(ValueError):bottom_neighbourhood([],None,None,v)
    def test_rotation_auto_set_before_metadata_read(self):
        class Capture:
            def get(self,prop):return 90
            def set(self,prop,value):self.option=(prop,value);return True
        c=Capture()
        with patch('squatvision.video.cv2.VideoCapture',return_value=c):
            self.assertIs(open_video('x'),c)
        self.assertEqual(c.option,(cv2.CAP_PROP_ORIENTATION_AUTO,1))
    def test_unsupported_rotation_fails_explicitly(self):
        c=SimpleNamespace(get=lambda _:90,set=lambda *_:False,release=lambda:None)
        with patch('squatvision.video.cv2.VideoCapture',return_value=c):
            with self.assertRaises(ValueError):open_video('x')

if __name__=='__main__':unittest.main()

class ReviewGuiTests(unittest.TestCase):
    def run_gui(self,keys,labels=None,proposal=None,visible=1):
        from manual_depth import collect_gui
        from contextlib import ExitStack
        self.positions=[]
        class Capture:
            def isOpened(self):return True
            def get(self,prop):return 120
            def set(inner,prop,value):self.positions.append(value)
            def read(self):return True,np.zeros((240,320,3),np.uint8)
            def release(self):pass
        with ExitStack() as stack:
            stack.enter_context(patch('manual_depth.open_video',return_value=Capture()))
            for name in ['namedWindow','setMouseCallback','destroyWindow']:
                stack.enter_context(patch('manual_depth.cv2.'+name))
            show=stack.enter_context(patch('manual_depth.cv2.imshow'))
            stack.enter_context(patch('manual_depth.cv2.getWindowProperty',return_value=visible))
            stack.enter_context(patch('manual_depth.cv2.waitKeyEx',side_effect=keys))
            result=collect_gui(Path('x'),30,320,240,labels or {},0,2,8,
                               {30:proposal} if proposal else {},30,1)
            return result,show.call_count
    def proposal(self):
        return dict(frame=30,hip={'point':[80,100]},knee={'point':[240,95]})
    def test_opens_at_reference_and_accepts_only_after_u(self):
        result,count=self.run_gui([ord('u'),ord('s')],proposal=self.proposal())
        self.assertEqual(self.positions[0],30);self.assertEqual(list(result),[30])
        self.assertEqual(result[30].source,MANUAL);self.assertGreaterEqual(count,2)
    def test_save_does_not_implicitly_accept_displayed_proposal(self):
        from squatvision.ipf_depth import ManualPoint
        existing={0:ManualPoint(0,(80,100),(240,95))}
        result,_=self.run_gui([ord('s')],existing,self.proposal())
        self.assertEqual(result,existing);self.assertNotIn(30,result)
    def test_window_close_cancels_without_reopening(self):
        result,count=self.run_gui([-1],proposal=self.proposal(),visible=0)
        self.assertIsNone(result);self.assertEqual(count,1)
    def test_partial_candidate_cannot_be_accepted(self):
        proposal=self.proposal();proposal['hip']['point']=None
        result,_=self.run_gui([ord('u'),27],proposal=proposal)
        self.assertIsNone(result)
