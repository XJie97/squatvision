"""Regression checks for review-only evidence and conservative decisions."""
import unittest
import cv2
import numpy as np

from squatvision.surface_evidence import (
    filter_temporal, preferred_review_frame, propose_surfaces, review_closeups)


class ReviewEvidenceUpdateTests(unittest.TestCase):
    def setUp(self):
        self.w, self.h = 320, 240
        self.image = np.full((self.h, self.w, 3), 85, np.uint8)
        self.points = ((.20,.2), (.25,.5), (.75,.5), (.80,.8))

    def make_pair(self, frame, margin=5, hip=True, knee=True):
        return {'frame':frame,'timestamp_s':frame/30,
                'status':'AUTOMATIC_CANDIDATE' if hip and knee else 'UNAVAILABLE',
                'margin_px':margin if hip and knee else None,
                'reason':'',
                'hip': {'point':[80.,100.] if hip else None,'offset_fraction':.32},
                'knee':{'point':[240.,95.] if knee else None,'offset_fraction':.3}}

    def test_unsupported_margin_is_raw_only(self):
        row=filter_temporal([self.make_pair(2188,-9.56)])[0]
        self.assertEqual(row['status'],'UNAVAILABLE')
        self.assertIsNone(row['margin_px'])
        self.assertEqual(row['raw_candidate_margin_px'],-9.56)
        self.assertEqual(row['reason'],'INSUFFICIENT_VALID_FRAMES')
        self.assertIsNotNone(row['hip']['point'])

    def test_supported_margin_remains_explicitly_unverified(self):
        rows=filter_temporal([self.make_pair(i) for i in range(5)])
        self.assertEqual(rows[2]['status'],'AUTOMATIC_UNVERIFIED')
        self.assertEqual(rows[2]['margin_px'],5)
        self.assertEqual(rows[2]['raw_candidate_margin_px'],5)

    def test_best_review_frame_prioritises_pair_not_arbitrary_knee_angle(self):
        rows=[self.make_pair(2187,hip=False,knee=True),
              self.make_pair(2188,-9.56),
              self.make_pair(2191,hip=True,knee=False)]
        selected, reason=preferred_review_frame(rows,2187)
        self.assertEqual(selected,2188)
        self.assertEqual(reason,'TWO_UNVERIFIED_POINTS_FOR_HUMAN_REVIEW')
        self.assertEqual(rows[1]['status'],'AUTOMATIC_CANDIDATE')

    def test_review_frame_handles_missing_evidence_without_fabrication(self):
        self.assertEqual(preferred_review_frame([],2187),(None,'NO_REVIEW_FRAMES'))
        rows=[self.make_pair(2187,hip=False,knee=False)]
        self.assertEqual(preferred_review_frame(rows,2187)[1],'NO_SURFACE_POINTS_AVAILABLE')

    def test_zoomed_closeups_preserve_source_and_are_readable(self):
        orig=self.image.copy()
        row=self.make_pair(2188)
        row['reason']='INSUFFICIENT_VALID_FRAMES'
        img=review_closeups(self.image,None,self.points,row)
        self.assertEqual(img.shape,(972,840,3))
        self.assertGreater(np.count_nonzero(img!=26),5000)
        np.testing.assert_array_equal(orig,self.image)

    def test_closeups_are_safe_at_frame_edge(self):
        pts=((.1,.1),(.01,.01),(.99,.02),(.8,.5))
        img=review_closeups(self.image,np.ones((self.h,self.w),np.float32),pts,
                            self.make_pair(0))
        self.assertEqual(img.shape,(972,840,3))

    def test_closeups_reject_bad_masks(self):
        with self.assertRaises(ValueError):
            review_closeups(self.image,np.zeros((10,10)),self.points,self.make_pair(0))

    def test_smooth_distal_silhouette_can_propose_curved_knee(self):
        # The knee cap has a *smooth curvature*, unlike a straight femoral
        # border. This is still only a visible image-edge proposal.
        mask=np.zeros((self.h,self.w),np.float32)
        for x in range(self.w):
            top = 100 + (0.03*(x-240)**2 if x > 215 else 0)
            if top < 140:
                mask[max(0,int(top)):min(self.h,145),x]=1.
        r=propose_surfaces(self.image,mask,self.points,self.w,self.h)
        self.assertIsNotNone(r['knee']['point'],r['knee']['reason'])
        self.assertIn(r['knee']['boundary_shape'],('linear','smooth_knee_curve'))

    def test_rgb_only_cannot_use_curved_silhouette_exception(self):
        mask=np.ones((self.h,self.w),np.float32)
        im=np.full_like(self.image,175)
        for x in range(self.w):
            top = 100 + (0.03*(x-240)**2 if x > 215 else 0)
            if top < self.h:
                im[int(top):,x,:]=30
        r=propose_surfaces(im,mask,self.points,self.w,self.h)
        # Not required to reject; if accepted, the result remains an
        # unverified clothing/colour-edge proxy, not semantic anatomy.
        if r['knee']['point'] is not None:
            self.assertEqual(r['knee']['method'],'internal_rgb_boundary')
            self.assertEqual(r['anatomical_identity'][:10],'UNVERIFIED')


if __name__=='__main__':
    unittest.main()
