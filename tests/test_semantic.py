import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from paper_priority.embeddings import mean_pool
from paper_priority.models import rows_to_x
from paper_priority.semantic_models import SemanticHead, HEADS


class SemanticTests(unittest.TestCase):
    def test_padding_excluded_and_unit_normalized(self):
        hidden = np.asarray([[[1., 0.], [0., 1.], [999., 999.]]])
        result = mean_pool(hidden, np.asarray([[1, 1, 0]]))
        np.testing.assert_allclose(result, [[2**-.5, 2**-.5]], atol=1e-7)

    def test_center_and_venue_fit_training_only(self):
        random = np.random.default_rng(261)
        embeddings = random.normal(size=(30, 12))
        rows = [dict(title='logic theory', venue='kr' if i < 25 else 'unseen', authors='A Person', year='2020')
                for i in range(30)]
        x, y = rows_to_x(rows), np.tile(np.arange(1, 6), 6)
        model = SemanticHead(HEADS['ordinal_meta_c4']).fit(embeddings[:25], x[:25], y[:25])
        np.testing.assert_array_equal(model.center_, embeddings[:25].mean(axis=0))
        self.assertNotIn('unseen', model.venue_.categories_[0])
        predictions = model.predict_components(embeddings[25:], x[25:])
        self.assertEqual(predictions.shape, (5, 2))
        self.assertTrue(np.all((predictions >= 0) & (predictions <= 1)))

    def test_each_head_round_trip_and_bounded_predictions(self):
        random = np.random.default_rng(261)
        embeddings = random.normal(size=(30, 12))
        x = rows_to_x([dict(title='logic theory', venue='kr', authors='A Person', year='2020') for _ in range(30)])
        y = np.tile(np.arange(1, 6), 6)
        with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as directory:
            for name, spec in HEADS.items():
                model = SemanticHead(spec).fit(embeddings, x, y)
                result = model.predict_components(embeddings, x)
                self.assertTrue(np.all(np.isfinite(result)))
                self.assertTrue(np.all((result >= 0) & (result <= 1)))
                path = Path(directory) / f'{name}.joblib'
                joblib.dump(model, path)
                np.testing.assert_array_equal(joblib.load(path).predict_components(embeddings, x), result)


if __name__ == '__main__':
    unittest.main()
