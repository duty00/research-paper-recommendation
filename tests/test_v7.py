import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from paper_priority.advanced import policy_priorities
from paper_priority.metadata_models import MetadataSVC
from paper_priority.models import rows_to_x
from paper_priority.v7_models import TunedMetadataSVC,priorities,candidates
from paper_priority.v7_experiment import confirmation


class V7Tests(unittest.TestCase):
    def test_scale_gamma_preserves_existing_metadata_classifier(self):
        rng=np.random.default_rng(261)
        e=rng.normal(size=(40,10))
        q=rng.normal(size=(5,10))
        x=rows_to_x([dict(title='logic theory',venue='kr',authors='Alice, Bob' if i%2 else '',year='2020') for i in range(40)])
        xt=rows_to_x([dict(title='new logic',venue='new_venue',authors='Unseen Author',year='2025') for _ in range(5)])
        y=np.tile(np.arange(1,6),8)
        spec=dict(encoder='bge',kind='svc',metadata=True,C=4.,authors=.75,venue=.35,numeric=.05)
        with threadpool_limits(limits=1):
            old=MetadataSVC(spec,261).fit(e,x,y)
            new=TunedMetadataSVC(dict(spec,gamma_factor=1.),261).fit(e,x,y)
            np.testing.assert_allclose(old.predict_components(q,xt),new.predict_components(q,xt),atol=1e-12,rtol=0)
            self.assertNotIn('unseen_author',new.authors_.vocabulary_)

    def test_half_weight_preserves_the_parent_selection_rule(self):
        ids=['9','10','2','1','8','3']
        values=np.asarray([[.2,.8],[.5,.5],[.2,.8],[.4,.4],[.6,.7],[.1,.2]])
        np.testing.assert_array_equal(priorities(ids,values,.6,.5),policy_priorities(ids,values,.6,'utility_half'))

    def test_ensembles_remain_convex(self):
        for spec in candidates().values():
            self.assertAlmostEqual(sum(spec['weights'].values()),1.)
            self.assertTrue(all(w>0 for w in spec['weights'].values()))

    def test_confirmation_rejects_one_seed_only_gain(self):
        base=dict(score=.6,fold_scores=[.6]*7)
        good=dict(score=.603,fold_scores=[.603]*7)
        bad=dict(score=.599,fold_scores=[.599]*7)
        self.assertTrue(confirmation(good,good,base,base)['eligible'])
        self.assertFalse(confirmation(good,bad,base,base)['eligible'])


if __name__=='__main__':
    unittest.main()
