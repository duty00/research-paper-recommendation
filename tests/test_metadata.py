import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from paper_priority.metadata_models import MetadataSVC
from paper_priority.models import rows_to_x


class MetadataTests(unittest.TestCase):
    def test_unseen_authors_and_venue_do_not_change_fitted_vocabulary(self):
        rng=np.random.default_rng(261)
        embedding=rng.normal(size=(30,12))
        train=rows_to_x([dict(title='logic theory',venue='kr',authors='Alice Smith, Bob Lee',year='2020') for _ in range(30)])
        test=rows_to_x([dict(title='new logic',venue='new_venue',authors='Unseen Author',year='2025')])
        spec=dict(kind='svc',metadata=True,C=4.,authors=.75,venue=.35,numeric=.05)
        with threadpool_limits(limits=1),tempfile.TemporaryDirectory() as directory:
            model=MetadataSVC(spec,261).fit(embedding,train,np.tile(np.arange(1,6),6))
            vocabulary=dict(model.authors_.vocabulary_)
            result=model.predict_components(embedding[:1],test)
            self.assertTrue(np.all((result>=0)&(result<=1)))
            self.assertEqual(vocabulary,model.authors_.vocabulary_)
            self.assertNotIn('unseen_author',vocabulary)
            file=Path(directory)/'model.joblib'
            joblib.dump(model,file)
            np.testing.assert_array_equal(joblib.load(file).predict_components(embedding[:1],test),result)

    def test_missing_authors_fit_without_inventing_an_author(self):
        rng=np.random.default_rng(261)
        x=rows_to_x([dict(title='logic theory',venue='kr',authors='',year='2020') for _ in range(30)])
        with threadpool_limits(limits=1):
            model=MetadataSVC(dict(kind='svc',metadata=True,C=4.,authors=.35,venue=.35,numeric=.05),261).fit(
                rng.normal(size=(30,12)),x,np.tile(np.arange(1,6),6))
            self.assertIsNone(model.authors_)


if __name__=='__main__':
    unittest.main()
