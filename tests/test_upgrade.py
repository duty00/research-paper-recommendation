import unittest
import numpy as np
from paper_priority.upgrade_experiment import BASE_WEIGHTS, candidates, folds_for, paired_check
from paper_priority.refine_models import aggregate


class UpgradeTests(unittest.TestCase):
    def test_candidate_preserves_exact_v5_mixture(self):
        rng = np.random.default_rng(261)
        predictions = {k:rng.uniform(size=(11,2)) for k in BASE_WEIGHTS}
        baseline = sum(BASE_WEIGHTS[k]*predictions[k] for k in BASE_WEIGHTS)
        for spec in candidates().values():
            self.assertAlmostEqual(sum(spec['weights'].values()), 1.)
            name = next(k for k in spec['weights'] if k not in BASE_WEIGHTS)
            q = spec['weights'][name]
            predictions[name] = rng.uniform(size=(11,2))
            np.testing.assert_allclose(aggregate(predictions, spec), (1-q)*baseline+q*predictions[name], atol=1e-15)

    def test_group_split_covers_each_row_and_never_splits_duplicates(self):
        x = np.arange(140).reshape(-1,1)
        y = np.tile(np.arange(1,6),28)
        groups = np.repeat(np.arange(70),2)
        folds, pairs = folds_for(x,y,groups,361)
        np.testing.assert_array_equal(np.sort(np.concatenate([v for _,v in pairs])), np.arange(140))
        for train,val in pairs:
            self.assertFalse(set(groups[train]) & set(groups[val]))
        np.testing.assert_array_equal(folds,folds_for(x,y,groups,361)[0])

    def test_gate_rejects_gain_only_on_first_seed_or_single_fold(self):
        base = dict(score=.6, fold_scores=[.6]*7)
        good = dict(score=.603, fold_scores=[.603]*7)
        self.assertTrue(paired_check(good,good,base,base)['eligible'])
        bad = dict(score=.599, fold_scores=[.599]*7)
        self.assertFalse(paired_check(good,bad,base,base)['eligible'])
        concentrated = dict(score=.61, fold_scores=[.67]+[.6]*6)
        self.assertFalse(paired_check(concentrated,concentrated,base,base)['eligible'])
        tiny = dict(score=.601, fold_scores=[.601]*7)
        self.assertFalse(paired_check(tiny,tiny,base,base)['eligible'])


if __name__ == '__main__':
    unittest.main()
