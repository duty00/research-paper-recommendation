import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from paper_priority.encoders_v4 import pool_hidden
from paper_priority.fusion_models import JointOrdinal
from paper_priority.models import rows_to_x


class FusionTests(unittest.TestCase):
    def test_bge_uses_cls_and_e5_uses_masked_mean(self):
        hidden=np.asarray([[[1.,0.],[0.,2.],[999.,999.]]])
        mask=np.asarray([[1,1,0]])
        np.testing.assert_allclose(pool_hidden(hidden,mask,'cls'),[[1.,0.]])
        np.testing.assert_allclose(pool_hidden(hidden,mask,'mean'),[[1/5**.5,2/5**.5]],atol=1e-7)

    def test_joint_fit_is_training_only_and_reproducible(self):
        rng=np.random.default_rng(261)
        embeddings=rng.normal(size=(25,8))
        rows=[dict(title='logic repeated theory',venue='kr',authors='A Person',year='2020') for _ in range(25)]
        rows[-1]['title']='secretvalidationtoken theory'
        x,y=rows_to_x(rows),np.tile(np.arange(1,6),5)
        with threadpool_limits(limits=1),tempfile.TemporaryDirectory() as directory:
            model=JointOrdinal(dict(profile='title',scale=2.,C=1.)).fit(embeddings[:20],x[:20],y[:20])
            np.testing.assert_array_equal(model.center_,embeddings[:20].mean(axis=0))
            self.assertNotIn('secretvalidationtoken',model.features_.named_transformers_['words'].vocabulary_)
            result=model.predict_components(embeddings[20:],x[20:])
            self.assertTrue(np.all((result>=0)&(result<=1)))
            path=Path(directory)/'model.joblib'
            joblib.dump(model,path)
            np.testing.assert_array_equal(joblib.load(path).predict_components(embeddings[20:],x[20:]),result)


if __name__=='__main__':
    unittest.main()
