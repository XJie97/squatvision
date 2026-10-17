"""Manual annotations must seed real visual tracking, not only exact-frame overlays."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from squatvision.analysis import FrameMetrics
from squatvision.optical_flow import calibration_from_manual_annotations, track_anatomy
from squatvision.ipf_depth import ManualPoint, apply_manual, measure, MANUAL, AUTOMATED


class TestAnnotationSeededTracking(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'moving.mp4'
        self.w, self.h=240,180
        fps=30
        rng=np.random.default_rng(7)
        patch_a=rng.integers(40,250,(42,42,3),dtype=np.uint8)
        patch_b=rng.integers(40,250,(42,42,3),dtype=np.uint8)
        self.n=19
        self.hips=[];self.knees=[]
        writer=cv2.VideoWriter(str(self.path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(self.w,self.h))
        assert writer.isOpened()
        for i in range(self.n):
            # Two independently moving features (hip right/down, knee left/down).
            hip=(54+i,70+i//3)
            knee=(169-i//2,119+i//5)
            self.hips.append(hip);self.knees.append(knee)
            frame=np.zeros((self.h,self.w,3),np.uint8)
            for (x,y),tex in ((hip,patch_a),(knee,patch_b)):
                frame[y-21:y+21,x-21:x+21]=tex
            writer.write(frame)
        writer.release()
        self.frames=[]
        for i in range(self.n):
            hip=self.hips[i];knee=self.knees[i]
            joints=((.19,.13),(hip[0]/self.w,hip[1]/self.h),
                    (knee[0]/self.w,knee[1]/self.h),(.8,.95))
            self.frames.append(FrameMetrics(i,i/fps,120,100,.4,.6,.9,'right',joints))
        self.manual={0:ManualPoint(0,self.hips[0],self.knees[0],6),
                     18:ManualPoint(18,self.hips[18],self.knees[18],6)}

    def test_two_anchors_produce_intermediate_independent_motion(self):
        cal, warning=calibration_from_manual_annotations(self.frames,self.manual,self.w,self.h)
        self.assertIsNotNone(cal)
        self.assertEqual(len(cal.additional_references),1)
        tracked=track_anatomy(self.path,self.frames,cal,self.w,self.h)
        valid=[p for p in tracked if p.crease_point and p.knee_top_point]
        self.assertGreaterEqual(len(valid),8)
        # The movement estimates are not copied from fixed X coordinates.
        for p in valid:
            i=p.frame
            hip_x=p.crease_point[0]*self.w;hip_y=p.crease_point[1]*self.h
            knee_x=p.knee_top_point[0]*self.w;knee_y=p.knee_top_point[1]*self.h
            self.assertLess(abs(hip_x-self.hips[i][0]),8.0)
            self.assertLess(abs(hip_y-self.hips[i][1]),8.0)
            self.assertLess(abs(knee_x-self.knees[i][0]),8.0)
            self.assertLess(abs(knee_y-self.knees[i][1]),8.0)
        # Manual frames stay manually sourced and remain exact after merge.
        merged=[apply_manual(f,self.manual[f.frame],self.w,self.h)
                if f.frame in self.manual else f for f in tracked]
        for i in (0,18):
            m=measure(merged[i],self.w,self.h,require_manual_verification=True)
            self.assertEqual(m.measurement_source,MANUAL)
            self.assertEqual(m.upper_thigh_at_hip_x,self.hips[i][0])
        automatic=[measure(f,self.w,self.h,require_manual_verification=True)
                   for f in merged[1:-1] if f.crease_point and f.knee_top_point]
        self.assertTrue(automatic)
        self.assertTrue(all(m.measurement_source==AUTOMATED for m in automatic))
        self.assertTrue(all(m.depth_status=='BORDERLINE_UNCERTAIN' for m in automatic))

    def test_invalid_pose_anchor_is_not_faked(self):
        all_bad=[FrameMetrics(f.frame,f.time_s) for f in self.frames]
        cal, warning=calibration_from_manual_annotations(all_bad,self.manual,self.w,self.h)
        self.assertIsNone(cal)
        self.assertIn('No usable manual anchors',warning)


if __name__=='__main__':
    unittest.main()

class TestFullExportWiring(unittest.TestCase):
    def test_manual_annotation_automatically_triggers_tracking(self):
        """Integration: CLI run must call tracking without --depth-calibration."""
        import json
        import sys
        import types
        from unittest.mock import patch
        import app
        from squatvision.analysis import Config
        from squatvision.ipf_depth import save_manual
        from test_app import StubLandmarker

        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            video=root/'clip.mp4'
            model=root/'model.task';model.write_bytes(b'test')
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(240,180))
            self.assertTrue(writer.isOpened())
            for _ in range(25):writer.write(np.full((180,240,3),100,np.uint8))
            writer.release()
            source=FrameMetrics(0,0,120,100,.4,.6,.9,"right",((.19,.13),(.225,.389),(.704,.661),(.8,.95)))
            # Every frame has a suitable pose; the real optical flow implementation
            # is substituted to isolate the main application's CLI wiring.
            def extract(_lm,i,t,_w,_h,_side,_v):
                return FrameMetrics(i,t,120,100,.4,.6,.9,'right',source.points)
            doc=root/'manual.json'
            save_manual(doc,'clip.mp4',240,180,30,[self.manual[0]]) if hasattr(self,'manual') else save_manual(
                doc,'clip.mp4',240,180,30,[ManualPoint(0,(54,70),(169,119),6)])
            fake_mp=types.SimpleNamespace(
                Image=lambda **kw:object(), ImageFormat=types.SimpleNamespace(SRGB=1),
                tasks=types.SimpleNamespace(BaseOptions=lambda **kw:object(),
                    vision=types.SimpleNamespace(RunningMode=types.SimpleNamespace(VIDEO=1),
                        PoseLandmarkerOptions=lambda **kw:object(),
                        PoseLandmarker=types.SimpleNamespace(create_from_options=lambda _:StubLandmarker()))))
            with patch.dict(sys.modules,{'mediapipe':fake_mp}),\
                 patch.object(app,'extract_frame',side_effect=extract),\
                 patch.object(app,'track_anatomy',side_effect=lambda v,f,c,w,h:f) as track:
                app.run(video,root/'out',model,'right',Config(),.5,0.,None,True,None,doc)
            self.assertEqual(track.call_count,1)
            summary=json.loads((root/'out'/'summary.json').read_text())
            self.assertTrue(summary['manual_seeded_visual_tracking'])
            self.assertFalse(summary['depth_calibration_applied'])
            self.assertEqual(summary['manual_annotation_count'],1)
