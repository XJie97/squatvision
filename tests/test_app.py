"""Full output-export test using a deterministic fake pose provider, not MediaPipe itself."""
import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import app
from squatvision.analysis import Config, FrameMetrics
from squatvision.depth import DepthCalibration
from squatvision.ipf_depth import ManualPoint, save_manual
from test_analysis import base_squat, create_video_sequence


class StubLandmarker:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def detect_for_video(self, _image, _timestamp):
        return types.SimpleNamespace(pose_landmarks=[[object()]])


class TestExports(unittest.TestCase):
    def test_video_and_timestamped_event_screenshots(self):
        frames=create_video_sequence(base_squat(pause_frames=20))
        # Landmarks chosen to make angle guides visible on the rendered frames.
        decorated=[FrameMetrics(f.frame, f.time_s, f.knee_deg, f.hip_deg,
                                f.hip_y, f.knee_y, f.visibility, 'right',
                                ((.4,.18),(.4,.41),(.52,.64),(.57,.88))) for f in frames]
        fake_mp=types.SimpleNamespace(
            Image=lambda **kw: object(),ImageFormat=types.SimpleNamespace(SRGB=7),
            tasks=types.SimpleNamespace(
                BaseOptions=lambda **kw: object(),
                vision=types.SimpleNamespace(
                    RunningMode=types.SimpleNamespace(VIDEO=1),
                    PoseLandmarkerOptions=lambda **kw: object(),
                    PoseLandmarker=types.SimpleNamespace(create_from_options=lambda _cfg: StubLandmarker()),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            input_video=directory/'sample.mp4'
            output=directory/'out'
            writer=cv2.VideoWriter(str(input_video),cv2.VideoWriter_fourcc(*'mp4v'),30.,(320,480))
            self.assertTrue(writer.isOpened())
            for i in range(len(frames)):
                raw=np.full((480,320,3),60,dtype=np.uint8)
                cv2.putText(raw,f'{i}',(40,240),cv2.FONT_HERSHEY_SIMPLEX,1,(220,220,220),2)
                writer.write(raw)
            writer.release()
            model=directory/'model.task'
            model.write_bytes(b'stub')
            def fake_extract(_landmarks,frame_index,_time,_w,_h,_side,_vis):
                return decorated[frame_index]
            with patch.dict(sys.modules, {'mediapipe':fake_mp}), \
                 patch.object(app,'extract_frame',side_effect=fake_extract):
                app.run(input_video,output,model,'right',Config(),.55,0.,None,False)
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['completed_reps'],1)
            self.assertEqual(summary['total_attempts'],1)
            self.assertTrue((output/'annotated.mp4').is_file())
            with (output/'events.csv').open() as f:
                events=list(csv.DictReader(f))
            self.assertEqual([e['event'] for e in events],
                             ['descent_start','descent_end','ascent_start','lockout'])
            for e in events:
                image=output/e['image']
                self.assertTrue(image.exists(), f'Missing {image}')
                self.assertIsNotNone(cv2.imread(str(image)))
                self.assertGreater(float(e['timestamp_s']),0.)
            # Texture-free synthetic video: optical flow MUST report unavailable
            # instead of fabricating depths from thigh pose geometry.
            reference=decorated[0]
            calibration=DepthCalibration.from_reference(
                reference, (reference.points[1][0]*320+8,reference.points[1][1]*480+14),
                (reference.points[2][0]*320-8,reference.points[2][1]*480-14),
                320,480)
            calibration_path=directory/'depth_calibration.json'
            calibration.save(calibration_path)
            calibrated_output=directory/'calibrated'
            with patch.dict(sys.modules, {'mediapipe':fake_mp}), \
                 patch.object(app,'extract_frame',side_effect=fake_extract):
                app.run(input_video,calibrated_output,model,'auto',Config(),.55,0.,None,
                        False,calibration_path)
            calibrated=json.loads((calibrated_output/'summary.json').read_text())
            self.assertTrue(calibrated['depth_calibration_applied'])
            with (calibrated_output/'events.csv').open() as f:
                calibrated_events=list(csv.DictReader(f))
            self.assertEqual(len(calibrated_events),4)
            for event in calibrated_events:
                self.assertEqual(event['crease_minus_knee_top_px'],'')
                self.assertEqual(event['crease_depth_status'],'unavailable')
                self.assertEqual(event['depth_method'],'local_optical_flow')
                self.assertTrue((calibrated_output/event['image']).is_file())
            with (calibrated_output/'frames.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertTrue(all(row['depth_status']=='unavailable' for row in rows))
            self.assertEqual(calibrated['per_attempt_depth']['1']['max_depth_status'],'unavailable')
            self.assertTrue((calibrated_output/'diagnostics'/'tracking_confidence.svg').exists())
            self.assertNotIn('depth_calibration.json', [p.name for p in calibrated_output.iterdir()])


class TestManualApplication(unittest.TestCase):
    def test_manual_depth_without_visual_tracking_in_main_pipeline(self):
        frames=create_video_sequence(base_squat(pause_frames=20))
        pose_frames=[FrameMetrics(f.frame,f.time_s,f.knee_deg,f.hip_deg,
                      f.hip_y,f.knee_y,f.visibility,'right',
                      ((.4,.18),(.4,.41),(.52,.64),(.57,.88))) for f in frames]
        fake_mp=types.SimpleNamespace(
            Image=lambda **kw: object(),ImageFormat=types.SimpleNamespace(SRGB=7),
            tasks=types.SimpleNamespace(BaseOptions=lambda **kw: object(),
                vision=types.SimpleNamespace(RunningMode=types.SimpleNamespace(VIDEO=1),
                    PoseLandmarkerOptions=lambda **kw: object(),
                    PoseLandmarker=types.SimpleNamespace(create_from_options=lambda _:StubLandmarker()))))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); video=root/'sample.mp4';out=root/'out'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(320,480))
            self.assertTrue(writer.isOpened())
            for _ in frames:writer.write(np.full((480,320,3),80,np.uint8))
            writer.release()
            model=root/'model.task';model.write_bytes(b'stub')
            # Frame 95 is near deepest knee flexion, while no optical flow is supplied.
            labels=root/'manual.json'
            save_manual(labels,'sample.mp4',320,480,30.,[
                ManualPoint(95,(110.,330.),(200.,310.),8.)])
            def extract(_l,i,_t,_w,_h,_side,_v):return pose_frames[i]
            with patch.dict(sys.modules,{'mediapipe':fake_mp}), \
                    patch.object(app,'extract_frame',side_effect=extract):
                app.run(video,out,model,'right',Config(),.55,0.,None,False,None,labels)
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['completed_reps'],1)
            self.assertTrue(summary['manual_depth_annotations_applied'])
            assessment=summary['ipf_depth_assessments'][0]['depth_assessment']
            self.assertEqual(assessment['depth_margin_px'],20.)
            self.assertEqual(assessment['measurement_source'],'MANUAL_ANNOTATION')
            with (out/'frames.csv').open() as f:
                rows={int(r['frame']):r for r in csv.DictReader(f)}
            self.assertEqual(rows[95]['ipf_depth_status'],'ESTIMATED_BELOW_DEPTH_THRESHOLD')
            self.assertEqual(rows[94]['ipf_depth_status'],'UNAVAILABLE')
            self.assertEqual(rows[95]['depth_margin_px'],'20.0')
            self.assertTrue((out/'depth_vs_time.svg').is_file())
            self.assertTrue((out/'depth_screenshots'/'manual_depth_000095.png').is_file())
            self.assertTrue((out/'annotated.mp4').is_file())


if __name__=='__main__':
    unittest.main()
