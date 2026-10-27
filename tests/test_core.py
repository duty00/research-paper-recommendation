import math
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from paper_priority.data import Dataset, clean_text, paper_groups, read_rows, write_csv
from paper_priority.experiment import predict_bundle, validate_submission
from paper_priority.metric import make_priorities, score_priorities
from paper_priority.models import fit_model, predict_gain, rows_to_x


class MetricTests(unittest.TestCase):
    def test_text_id_tie_break_and_hand_calculation(self):
        # Text order is '10' before '2', even though numeric order is different.
        metric = score_priorities(['2', '10'], [1, 5], [1, 1])
        expected_q = 1 / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(metric['relevance'], expected_q)
        self.assertEqual(metric['coverage'], 1)
        self.assertAlmostEqual(metric['score'], 2 * expected_q / (expected_q + 1))

    def test_zero_exclusion_and_all_positive_inclusion(self):
        metric = score_priorities(['a', 'b', 'c'], [5, 3, 1], [2, 0, 1e-12])
        self.assertEqual(metric['recommended'], 2)
        self.assertEqual(metric['coverage'], 0.5)
        self.assertLess(metric['relevance'], 1)

    def test_empty_and_no_worthwhile(self):
        self.assertEqual(score_priorities(['a'], [5], [0])['score'], 0)
        self.assertEqual(score_priorities(['a'], [1], [1])['score'], 0)

    def test_reject_invalid(self):
        for priority in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                score_priorities(['a'], [5], [priority])
        with self.assertRaises(ValueError):
            score_priorities(['a', 'a'], [1, 2], [1, 2])

    def test_rank_encoding_and_selection(self):
        ids = ['2', '10', '3', '4']
        gain = [0.8, 0.8, 0.6, 0.2]
        p = make_priorities(ids, gain, 0.5)
        self.assertEqual(list(p), [5.0, 10.0, 0.0, 0.0])
        reversed_score = score_priorities(ids[::-1], [1, 5, 3, 2][::-1], p[::-1])
        self.assertEqual(reversed_score, score_priorities(ids, [1, 5, 3, 2], p))


class DataTests(unittest.TestCase):
    def test_rtf_wrapped_csv_and_unicode(self):
        raw = ('{\\rtf1\\ansi\\ansicpg1252\n'
               '\\f0\\fs24 \\cf0 id,title,venue,year,authors,doi\\\n'
               '979,A &amp; B,kr,2020,M\\u252?ller,10.1/example\\\n'
               '2635,Last paper,lics,2021,,https://example.test/paper}')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'Test.csv'
            path.write_text(raw, encoding='utf-8')
            rows, repaired = read_rows(path)
        self.assertTrue(repaired)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['authors'], 'Müller')
        self.assertEqual(clean_text(rows[0]['title']), 'A & B')
        self.assertEqual(rows[1]['doi'], 'https://example.test/paper')

    def test_duplicate_groups_are_transitive(self):
        rows = [dict(title='A paper!', doi='10/a'), dict(title='a paper', doi='10/b'),
                dict(title='Another paper', doi='https://doi.org/10/b'),
                dict(title='Different', doi='')]
        groups = paper_groups(rows)
        self.assertEqual(len(set(groups[:3])), 1)
        self.assertNotEqual(groups[2], groups[3])

    def test_no_group_crosses_validation_boundary(self):
        groups = np.repeat(np.arange(30), 2)
        labels = np.repeat(np.arange(30) % 5 + 1, 2)
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=261)
        for fit, val in splitter.split(np.zeros((60, 1)), labels, groups):
            self.assertFalse(set(groups[fit]) & set(groups[val]))

    def test_submission_keys_and_finite_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'submission.csv'
            for ids, priorities in [(['a', 'a'], [1, 2]), (['a', 'x'], [1, 2]),
                                     (['a', 'b'], [1, float('nan')])]:
                write_csv(path, [dict(id=i, priority=p) for i, p in zip(ids, priorities)], ['id', 'priority'])
                with self.assertRaises(ValueError):
                    validate_submission(path, ['a', 'b'])
            write_csv(path, [dict(id='b', priority=0), dict(id='a', priority=2)], ['id', 'priority'])
            self.assertEqual(validate_submission(path, ['a', 'b'])['recommended'], 1)

    def test_prediction_cannot_overwrite_sources(self):
        source = Path('original.csv').resolve()
        data = Dataset([], [], [], {'test': source}, {})
        with self.assertRaisesRegex(ValueError, 'overwrite an original'):
            predict_bundle(Path('unused_model.joblib'), data, source)


class ModelTests(unittest.TestCase):
    def test_predictors_exclude_ids_and_doi(self):
        row = dict(id='1', title='Reasoning with logic', venue='kr', year='2020', authors='', doi='secret')
        changed = row | dict(id='999', doi='other')
        np.testing.assert_array_equal(rows_to_x([row]), rows_to_x([changed]))

    def test_vocab_train_only_and_model_round_trip(self):
        rows = [dict(id=str(i), title=('logic planning knowledge' if i % 2 else 'program verification systems'),
                     venue='kr' if i % 2 else 'cav', year='2020', authors='', doi='') for i in range(12)]
        labels = np.asarray([5 if i % 2 else 1 for i in range(12)])
        test_rows = [rows[0] | dict(title='unseenvalidationtoken planning', venue='newvenue')]
        for name in ('ridge_a1', 'logistic_c1'):
            model = fit_model(name, rows_to_x(rows), labels, 261)
            vocabulary = model.named_steps['features'].named_transformers_['words'].vocabulary_
            self.assertNotIn('unseenvalidationtoken', vocabulary)
            before = predict_gain(model, rows_to_x(test_rows))
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'model.joblib'
                joblib.dump(model, path)
                after = predict_gain(joblib.load(path), rows_to_x(test_rows))
            np.testing.assert_array_equal(before, after)


if __name__ == '__main__':
    unittest.main()
