import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from paper_priority.author_models import AuthorPreference,HEADS
from paper_priority.models import rows_to_x


class AuthorPreferenceTests(unittest.TestCase):
    def test_each_family_handles_unseen_names_and_round_trips(self):
        x=rows_to_x([dict(title='logic',venue='kr' if i%2 else 'cav',authors='Alice, Bob' if i%3 else '',year='2020') for i in range(40)])
        query=rows_to_x([dict(title='new paper',venue='new_venue',authors='Unseen Name',year='2025')])
        y=np.tile(np.arange(1,6),8)
        with threadpool_limits(limits=1),tempfile.TemporaryDirectory() as directory:
            for name in ('author_ridge_a10','author_ordinal_c1','author_prior_s2'):
                model=AuthorPreference(HEADS[name]).fit(x,y)
                values=model.predict_components(query)
                self.assertEqual(values.shape,(1,2))
                self.assertTrue(np.all((values>=0)&(values<=1)))
                if model.spec['kind']=='prior':
                    self.assertNotIn('unseen_name',model.offsets_)
                    np.testing.assert_allclose(values[0],model.global_)
                else:
                    self.assertNotIn('unseen_name',model.authors_.vocabulary_)
                path=Path(directory)/f'{name}.joblib'
                joblib.dump(model,path)
                np.testing.assert_array_equal(joblib.load(path).predict_components(query),values)

    def test_prior_shrinks_rare_authors_toward_venue_preference(self):
        x=rows_to_x([dict(title='p',venue='kr',authors='Rare' if i==0 else '',year='2020') for i in range(10)])
        y=np.asarray([5]+[1]*9)
        query=x[:1]
        weak=AuthorPreference(dict(kind='prior',smoothing=2.)).fit(x,y)
        strong=AuthorPreference(dict(kind='prior',smoothing=100.)).fit(x,y)
        self.assertGreater(weak.predict_components(query)[0,0],strong.predict_components(query)[0,0])


if __name__=='__main__':
    unittest.main()
