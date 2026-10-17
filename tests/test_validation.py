import unittest
from validate_depth import measure

class TestEvaluation(unittest.TestCase):
    def test_missing_and_valid_observations(self):
        preds=[{'frame':'1','hip_crease_x':'100','hip_crease_y':'120',
                'knee_top_x_px':'150','knee_top_y_px':'145','depth_difference_px':'-25'},
               {'frame':'2','hip_crease_x':'','hip_crease_y':'',
                'knee_top_x_px':'','knee_top_y_px':'','depth_difference_px':''}]
        refs=[{'frame':'1','hip_crease_x':'103','hip_crease_y':'124',
               'knee_top_x':'150','knee_top_y':'143'},
              {'frame':'2','hip_crease_x':'104','hip_crease_y':'128',
               'knee_top_x':'151','knee_top_y':'144'}]
        report,entries=measure(preds,refs)
        self.assertEqual(report['measurement_coverage'],.5)
        self.assertEqual(report['hip_crease_error_px']['mean'],5.)
        self.assertEqual(entries[0]['depth_error_px'],6.)

if __name__=='__main__':unittest.main()
