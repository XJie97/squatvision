"""Regression for real-video hip crease loss at 72.724 -> 72.758 seconds.

Six coherent inlier features were previously given confidence 6/14=0.429,
which passed the tracker's 0.4 threshold but failed its consumer's 0.5
threshold. The marker disappeared even though the point was tracked.
"""
import unittest
from unittest.mock import patch
import numpy as np
from squatvision.optical_flow import LocalPointTracker


class TestSparseFeatureConfidence(unittest.TestCase):
    def test_six_coherent_features_can_be_exported(self):
        gray=np.full((160,200),100,np.uint8)
        pts=np.array([(90,90),(94,96),(102,86),(105,95),(95,103),(99,110)],np.float32)
        def good_init(self, image, xy):
            return pts.copy()
        call=[0]
        def flow(_before,_after,points,_unused,**kwargs):
            call[0]+=1
            offset=np.array([1.,0.6],np.float32)
            shifted=points+offset if call[0]%2 else points-offset
            return shifted, np.ones((6,1),np.uint8), np.zeros((6,1),np.float32)
        def affine(_src,_dst,**kwargs):
            return np.array([[1.,0.,1.],[0.,1.,0.6]],np.float64), np.ones((6,1),np.uint8)
        with patch.object(LocalPointTracker,'_features',good_init), \
             patch('cv2.calcOpticalFlowPyrLK',side_effect=flow), \
             patch('cv2.estimateAffinePartial2D',side_effect=affine):
            tr=LocalPointTracker(gray,(100.,100.))
            self.assertEqual(len(tr.features),6)
            tr.update(gray,pose_expected=(101.,100.6),max_pose_disagreement=12.)
            self.assertTrue(tr.active)
            self.assertGreaterEqual(tr.confidence,.5)
            self.assertAlmostEqual(float(tr.xy[0]),101.,places=3)
            self.assertAlmostEqual(float(tr.xy[1]),100.6,places=3)


if __name__ == '__main__':
    unittest.main()
