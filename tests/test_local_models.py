import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from paper_priority.local_models import HEADS,LocalSemanticHead
from paper_priority.models import rows_to_x


class LocalSemanticTests(unittest.TestCase):
    def test_unseen_venue_constant_targets_and_roundtrip(self):
        rng=np.random.default_rng(261)
        e=rng.normal(size=(20,8));q=rng.normal(size=(3,8))
        x=rows_to_x([dict(title='train',venue='kr',year='2020',authors='') for _ in range(20)])
        xt=rows_to_x([dict(title='query',venue='unseen',year='2025',authors='New Person') for _ in range(3)])
        with threadpool_limits(limits=1),tempfile.TemporaryDirectory() as directory:
            for name in ('bge_krr_g1_a1','bge_neighbors30'):
                model=LocalSemanticHead(HEADS[name]).fit(e,x,np.full(20,3))
                result=model.predict_components(q,xt)
                np.testing.assert_allclose(result,np.tile([.5,1.],(3,1)),atol=1e-12)
                np.testing.assert_allclose(model.center_,e.mean(axis=0))
                self.assertNotIn('unseen',model.venue_.categories_[0])
                path=Path(directory)/'model.joblib';joblib.dump(model,path)
                np.testing.assert_array_equal(joblib.load(path).predict_components(q,xt),result)

    def test_neighbors_match_a_manual_weighted_average(self):
        e=np.asarray([[0.,0.],[1.,0.],[0.,1.]])
        x=rows_to_x([dict(title='t',venue='kr',year='2020',authors='') for _ in range(3)])
        spec=dict(kind='neighbors',neighbors=3,temperature=.15,smoothing=5.)
        model=LocalSemanticHead(spec).fit(e,x,np.array([1,3,5]))
        f=model.transform(e[:1],x[:1]);dist=np.sum((model.features_-f)**2,axis=1)
        w=np.exp(-(dist-dist.min())/.3)
        expected=(w@model.targets_+5*model.prior_)/(w.sum()+5)
        np.testing.assert_allclose(model.predict_components(e[:1],x[:1])[0],expected)

    def test_both_families_preserve_probability_bounds(self):
        rng=np.random.default_rng(1);e=rng.normal(size=(25,8));q=rng.normal(size=(4,8))
        x=rows_to_x([dict(title='t',venue='kr',year='2020',authors='') for _ in range(25)])
        with threadpool_limits(limits=1):
            for name in ('bge_krr_g2_a0.1','bge_neighbors30'):
                model=LocalSemanticHead(HEADS[name]).fit(e,x,np.tile(np.arange(1,6),5))
                result=model.predict_components(q,x[:4])
                self.assertTrue(np.isfinite(result).all())
                self.assertTrue(((result>=0)&(result<=1)).all())
                self.assertTrue((result[:,0]<=result[:,1]).all())
