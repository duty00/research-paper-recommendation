import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from paper_priority.refine_models import aggregate,RefinedSVC
from paper_priority.models import rows_to_x


class RefinementTests(unittest.TestCase):
    def test_rank_ensemble_is_invariant_to_positive_affine_calibration(self):
        a=np.asarray([[.1,.2],[.5,.8],[.3,.6],[.5,.7]])
        b=np.asarray([[.7,.8],[.1,.2],[.2,.3],[.4,.6]])
        spec=dict(weights={'a':.5,'b':.5},aggregation='rank')
        before=aggregate(dict(a=a,b=b),spec)
        changed=a.copy()
        changed[:,0]=changed[:,0]*7+3
        np.testing.assert_array_equal(aggregate(dict(a=changed,b=b),spec),before)
        permutation=np.asarray([3,0,2,1])
        after=aggregate(dict(a=a[permutation],b=b[permutation]),spec)
        np.testing.assert_array_equal(after,before[permutation])

    def test_rank_ensemble_handles_ties_and_single_item(self):
        spec=dict(weights={'a':1.},aggregation='rank')
        result=aggregate(dict(a=np.asarray([[.5,.8],[.5,.7]])),spec)
        np.testing.assert_array_equal(result[:,0],[.5,.5])
        single=aggregate(dict(a=np.asarray([[.5,.8]])),spec)
        self.assertTrue(np.all(np.isfinite(single)))

    def test_refined_classifier_round_trip_with_unseen_queries(self):
        rng=np.random.default_rng(261)
        embeddings=rng.normal(size=(30,12))
        query=rng.normal(size=(3,12))
        x=rows_to_x([dict(title='logic theory',venue='kr',authors='',year='2020') for _ in range(30)])
        y=np.tile(np.arange(1,6),6)
        spec=dict(kind='svc',metadata=False,C=16.,gamma_factor=.5)
        with threadpool_limits(limits=1),tempfile.TemporaryDirectory() as directory:
            model=RefinedSVC(spec,261).fit(embeddings,x,y)
            result=model.predict_components(query,x[:3])
            self.assertEqual(result.shape,(3,2))
            self.assertTrue(np.all((result>=0)&(result<=1)))
            path=Path(directory)/'model.joblib'
            joblib.dump(model,path)
            np.testing.assert_array_equal(joblib.load(path).predict_components(query,x[:3]),result)


if __name__=='__main__':
    unittest.main()
