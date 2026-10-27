"""Tests for the v2 decision policy and supervised feature fitting."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from paper_priority.advanced import AdvancedModel, policy_priorities
from paper_priority.models import rows_to_x, fit_model, predict_gain


class AdvancedTests(unittest.TestCase):
    def test_recall_selection_still_orders_by_gain(self):
        values = np.asarray([[.8, .7], [.3, .99], [.6, .9]])
        priority = policy_priorities(['a', 'b', 'c'], values, 2/3, 'read')
        self.assertEqual(priority[0], 0)
        self.assertGreater(priority[2], priority[1])

    def test_policy_ties_use_text_ids(self):
        priority = policy_priorities(['20', '3', '10'], np.full((3, 2), .5), 2/3, 'utility_one')
        self.assertGreater(priority[2], priority[0])
        self.assertEqual(priority[1], 0)

    def test_invalid_components_rejected(self):
        for values in ([[np.nan, .5]], [[.5, 1.01]], [[.5]]):
            with self.assertRaises(ValueError):
                policy_priorities(['a'], values, 1, 'gain')

    def test_folds_fit_ordinal_and_nb_without_test_vocabulary(self):
        rows = [dict(title=f'logic repeated theory {i%5}', venue='kr', authors='A Person', year='2020')
                for i in range(20)]
        x = rows_to_x(rows)
        query = rows_to_x([dict(title='secretunseen repeated theory', venue='kr', authors='A Person', year='2021')])
        y = np.tile(np.arange(1, 6), 4)
        with threadpool_limits(limits=1):
            for profile, kind in [('title', 'ordinal'), ('counts', 'nb_ordinal'), ('title', 'hurdle')]:
                model = AdvancedModel(dict(profile=profile, kind=kind, C=1, alpha=5)).fit(x, y)
                result = model.predict_components(query)
                self.assertEqual(result.shape, (1, 2))
                self.assertTrue(np.all((result >= 0) & (result <= 1)))
                if profile == 'counts':
                    vocab = model.features_.named_steps['counts'].vocabulary_
                else:
                    vocab = model.features_.named_transformers_['words'].vocabulary_
                self.assertNotIn('secretunseen', vocab)
                with tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / 'bundle.joblib'
                    joblib.dump(model, path)
                    np.testing.assert_array_equal(joblib.load(path).predict_components(query), result)

    def test_v1_components_match_existing_model(self):
        rows = [dict(title=f'logic repeated theory {i%5}', venue='kr', authors='A Person', year='2020')
                for i in range(20)]
        x, y = rows_to_x(rows), np.tile(np.arange(1, 6), 4)
        with threadpool_limits(limits=1):
            old = fit_model('logistic_c1', x, y, 261)
            new = AdvancedModel(dict(profile='v1', kind='multiclass', C=1), 261).fit(x, y)
            np.testing.assert_array_equal(new.predict_components(x)[:, 0], predict_gain(old, x))


if __name__ == '__main__':
    unittest.main()
