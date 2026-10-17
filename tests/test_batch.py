"""Multi-video batch processing, isolated per-video configs and CLI compatibility."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
from squatvision.batch import BatchInputError, find_mp4_files, run_batch


class BatchTests(unittest.TestCase):
    def test_sorted_mp4_detection_and_ignore_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('rear.mp4', 'front.MP4', 'side.mp4', 'notes.txt'):
                (root / name).write_bytes(b'fake video')
            (root / 'nested').mkdir()
            (root / 'nested' / 'hidden.mp4').touch()
            self.assertEqual([p.name for p in find_mp4_files(root)],
                             ['front.MP4', 'rear.mp4', 'side.mp4'])

    def test_no_files_and_nonexistent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(BatchInputError, 'No .mp4'):
                find_mp4_files(root)
            with self.assertRaisesRegex(BatchInputError, 'does not exist'):
                find_mp4_files(root / 'missing')

    def test_windows_case_insensitive_output_collision(self):
        # On normal Windows filesystems, these two .touch() calls refer to the
        # SAME file and do not create two distinct entries. Simulate a directory
        # listing containing both names so the collision check is tested on
        # Windows, Linux and macOS instead of depending on filesystem settings.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = [root / 'View.mp4', root / 'view.MP4']
            with patch.object(Path, 'iterdir', return_value=iter(entries)), \
                 patch.object(Path, 'is_file', return_value=True):
                with self.assertRaisesRegex(BatchInputError, 'same output directory'):
                    find_mp4_files(root)

    def test_per_video_output_summary_and_continue_on_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vids = root / 'data'; vids.mkdir()
            out = root / 'outputs'
            for name in ('front.mp4', 'rear.mp4', 'side.mp4'):
                (vids / name).touch()
            invoked = []
            def analyse(video, dest, annotations, calibration):
                self.assertIsNone(annotations)
                self.assertIsNone(calibration)
                invoked.append(video.name)
                if video.name == 'rear.mp4':
                    raise RuntimeError('synthetic decoder failure')
                dest.mkdir(parents=True)
                (dest / 'annotated.mp4').write_bytes(b'fake annotated file')
                (dest / 'summary.json').write_text(json.dumps({
                    'completed_reps': 1,
                    'total_attempts': 1,
                    'selected_side': 'right',
                    'usable_pose_fraction': .98,
                    'ipf_depth_assessments': [
                        {'depth_assessment': {'status': 'UNAVAILABLE'}}]
                }))
            rows = run_batch(vids, out, analyse)
            self.assertEqual(invoked, ['front.mp4', 'rear.mp4', 'side.mp4'])
            self.assertEqual([x['status'] for x in rows],
                             ['success', 'failed', 'success'])
            self.assertEqual(rows[1]['error'], 'synthetic decoder failure')
            self.assertTrue((out / 'front' / 'annotated.mp4').exists())
            self.assertTrue((out / 'side' / 'annotated.mp4').exists())
            self.assertEqual(rows[0]['depth_statuses'], 'UNAVAILABLE')
            self.assertEqual(json.loads((out / 'batch_summary.json').read_text()), rows)
            with (out / 'batch_summary.csv').open() as f:
                csv_rows = list(csv.DictReader(f))
            self.assertEqual(len(csv_rows), 3)
            self.assertEqual(csv_rows[1]['status'], 'failed')

    def test_annotations_are_video_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            videos=root/'data'; videos.mkdir()
            manual=root/'manual'; manual.mkdir()
            calibration=root/'cal'; calibration.mkdir()
            for s in ('front','side'):
                (videos / (s+'.mp4')).touch()
            (manual / 'front.json').write_text('{}')
            (calibration / 'side.json').write_text('{}')
            observed=[]
            def analyse(vid, dest, annotation, cal):
                observed.append((vid.stem,
                                 annotation.name if annotation else None,
                                 cal.name if cal else None))
                dest.mkdir(parents=True)
                (dest/'summary.json').write_text('{}')
            run_batch(videos,root/'out',analyse,manual,calibration)
            self.assertEqual(observed,[('front','front.json',None),('side',None,'side.json')])

    def test_reject_output_same_as_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'example.mp4').touch()
            with self.assertRaisesRegex(BatchInputError,'differ'):
                run_batch(root,root,lambda *args:None)

    def test_cli_default_batch_and_single_video(self):
        # The CLI must call the batch helper when no --input is given,
        # and continue to support --input for legacy use.
        with patch.object(app,'run_batch',return_value=[{'status':'success'}]) as batch, \
             patch.object(sys,'argv',['app.py']):
            app.main()
            self.assertEqual(str(batch.call_args.args[0]),'data')
            self.assertEqual(str(batch.call_args.args[1]),'outputs')
        with patch.object(app,'run') as single, \
             patch.object(sys,'argv',['app.py','--input','data/squat.mp4']):
            app.main()
            self.assertEqual(single.call_args.args[0],Path('data/squat.mp4'))
            self.assertEqual(single.call_args.args[1],Path('outputs/demo'))
        with patch.object(app,'run_batch',return_value=[{'status':'failed'}]), \
             patch.object(sys,'argv',['app.py']):
            with self.assertRaises(SystemExit) as err:
                app.main()
            self.assertEqual(err.exception.code,1)

    def test_cli_rejects_global_calibration_for_batch(self):
        with patch.object(sys,'argv',['app.py','--manual-annotations','data/manual_depth.json']):
            with self.assertRaises(SystemExit) as err:
                app.main()
            self.assertEqual(err.exception.code,2)


if __name__ == '__main__':
    unittest.main()
