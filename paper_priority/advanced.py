"""Version 2 CPU models. All supervised transforms are fitted inside training folds."""
from __future__ import annotations

import re
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, OneHotEncoder, normalize
from sklearn.svm import SVC

from .models import as_float, build_model

MODEL_SPECS = {
    'v1_logistic': dict(profile='v1', kind='multiclass', C=1.0),
    'multi_title_c1': dict(profile='title', kind='multiclass', C=1.0),
    'multi_rich_c1': dict(profile='rich', kind='multiclass', C=1.0),
    'multi_balanced_c05': dict(profile='balanced', kind='multiclass', C=0.5),
    'multi_rich_c4': dict(profile='rich', kind='multiclass', C=4.0),
    'ordinal_title_c1': dict(profile='title', kind='ordinal', C=1.0),
    'ordinal_balanced_c1': dict(profile='balanced', kind='ordinal', C=1.0),
    'ordinal_rich_c2': dict(profile='rich', kind='ordinal', C=2.0),
    'nb_ordinal_c1': dict(profile='counts', kind='nb_ordinal', C=1.0),
    'nb_ordinal_c4': dict(profile='counts', kind='nb_ordinal', C=4.0),
    'hurdle_balanced': dict(profile='balanced', kind='hurdle', C=1.0, alpha=5.0),
    'hurdle_title': dict(profile='title', kind='hurdle', C=1.0, alpha=5.0),
    'svd_svc_title': dict(profile='title', kind='svd_svc', C=4.0, dimensions=96),
}
CANDIDATE_WEIGHTS = {name: {name: 1.0} for name in MODEL_SPECS}
CANDIDATE_WEIGHTS.update({
    'blend_v1_rich': {'v1_logistic': .5, 'multi_rich_c1': .5},
    'blend_ordinal_rich': {'ordinal_rich_c2': .5, 'multi_rich_c1': .5},
    'blend_v1_nb': {'v1_logistic': .75, 'nb_ordinal_c1': .25},
})
POLICIES = ('gain', 'read', 'utility_half', 'utility_one')
FRACTIONS = (.60, .65, .70, .75, .80, .85, .90, .95, 1.00)


def author_tokens(text):
    return [re.sub(r'\s+', '_', name.strip().casefold()) for name in text.split(',') if name.strip()]


def title_column(x):
    return np.asarray(x, dtype=object)[:, 0]


def make_features(profile):
    if profile == 'v1':
        return build_model('logistic_c1', 261).named_steps['features']
    if profile == 'counts':
        return Pipeline([
            ('title', FunctionTransformer(title_column)),
            ('counts', CountVectorizer(ngram_range=(1, 2), binary=True, min_df=2,
                                       max_features=30000, strip_accents='unicode'))])
    rich = profile == 'rich'
    transformers = [
        ('words', TfidfVectorizer(ngram_range=(1, 3) if rich else (1, 2), min_df=1,
                                  max_features=35000, sublinear_tf=True, strip_accents='unicode'), 0),
        ('chars', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2,
                                 max_features=40000, sublinear_tf=True), 0),
    ]
    weights = {'words': 1.0, 'chars': .8}
    if profile != 'title':
        transformers.extend([
            ('venue', OneHotEncoder(handle_unknown='ignore'), [1]),
            ('authors', TfidfVectorizer(tokenizer=author_tokens, token_pattern=None,
                                        lowercase=False, min_df=2, max_features=6000), 2),
            ('numeric', FunctionTransformer(as_float), [3, 4, 5, 6]),
        ])
        weights.update(venue=.15, authors=.2 if rich else .1, numeric=.1 if rich else .05)
    return ColumnTransformer(transformers, transformer_weights=weights, sparse_threshold=1.0)


def fit_binary(x, target, C, seed):
    if len(np.unique(target)) < 2:
        return DummyClassifier(strategy='prior').fit(x, target)
    return LogisticRegression(C=C, solver='lbfgs', max_iter=2000,
                              tol=1e-5, random_state=seed).fit(x, target)


def positive_probability(estimator, x):
    if 1 not in estimator.classes_:
        return np.zeros(x.shape[0])
    return estimator.predict_proba(x)[:, list(estimator.classes_).index(1)]


class AdvancedModel(BaseEstimator):
    """Predict two components: expected gain and probability that rating > 1."""
    def __init__(self, spec, seed=261):
        self.spec = spec
        self.seed = seed

    def fit(self, x, y):
        features = make_features(self.spec['profile'])
        matrix = features.fit_transform(x)
        return self.fit_transformed(features, matrix, y)

    def fit_transformed(self, features, matrix, y):
        self.features_ = features
        y = np.asarray(y, dtype=int)
        kind, C = self.spec['kind'], self.spec['C']
        if kind == 'multiclass':
            self.estimator_ = LogisticRegression(C=C, solver='lbfgs', max_iter=1500,
                                                  tol=1e-5, random_state=self.seed).fit(matrix, y)
        elif kind in ('ordinal', 'nb_ordinal'):
            self.estimators_, self.ratios_ = [], []
            for boundary in (1, 2, 3, 4):
                target = (y > boundary).astype(int)
                if kind == 'nb_ordinal':
                    if matrix.nnz and np.min(matrix.data) < 0:
                        raise ValueError('NB features require nonnegative input counts.')
                    pos = 1 + np.asarray(matrix[target == 1].sum(axis=0)).ravel()
                    neg = 1 + np.asarray(matrix[target == 0].sum(axis=0)).ravel()
                    ratio = np.log(pos / pos.sum()) - np.log(neg / neg.sum())
                    weighted = normalize(matrix.multiply(ratio).tocsr())
                    self.ratios_.append(ratio)
                else:
                    weighted = matrix
                self.estimators_.append(fit_binary(weighted, target, C, self.seed))
        elif kind == 'hurdle':
            self.read_model_ = fit_binary(matrix, (y > 1).astype(int), C, self.seed)
            useful = y > 1
            if not np.any(useful):
                self.gain_model_ = None
            else:
                self.gain_model_ = Ridge(alpha=self.spec['alpha'], solver='lsqr', tol=1e-6).fit(
                    matrix[useful], (y[useful] - 1) / 4)
        elif kind == 'svd_svc':
            dimensions = min(self.spec['dimensions'], matrix.shape[0] - 1, matrix.shape[1] - 1)
            if dimensions < 1:
                raise ValueError('Not enough data for SVD.')
            self.reduction_ = Pipeline([
                ('svd', TruncatedSVD(n_components=dimensions, n_iter=7, random_state=self.seed)),
                ('normalize', Normalizer())])
            dense = self.reduction_.fit_transform(matrix)
            self.estimator_ = SVC(C=C, kernel='rbf', probability=True, random_state=self.seed).fit(dense, y)
        else:
            raise ValueError(f'Unknown model kind: {kind}')
        return self

    def predict_components(self, x):
        return self.predict_components_transformed(self.features_.transform(x))

    def predict_components_transformed(self, matrix):
        kind = self.spec['kind']
        if kind in ('multiclass', 'svd_svc'):
            if kind == 'svd_svc':
                matrix = self.reduction_.transform(matrix)
            probabilities = self.estimator_.predict_proba(matrix)
            classes = self.estimator_.classes_.astype(int)
            gain = probabilities @ ((classes - 1) / 4)
            readable = 1 - probabilities[:, list(classes).index(1)] if 1 in classes else np.ones(matrix.shape[0])
        elif kind in ('ordinal', 'nb_ordinal'):
            probabilities = []
            for index, estimator in enumerate(self.estimators_):
                weighted = normalize(matrix.multiply(self.ratios_[index]).tocsr()) if kind == 'nb_ordinal' else matrix
                probabilities.append(positive_probability(estimator, weighted))
            cumulative = np.column_stack(probabilities)
            # E[(Y-1)/4] = mean(P(Y>1), ..., P(Y>4)). These are separate binary estimates.
            gain, readable = cumulative.mean(axis=1), cumulative[:, 0]
        else:
            readable = positive_probability(self.read_model_, matrix)
            conditional = np.clip(self.gain_model_.predict(matrix), .25, 1) if self.gain_model_ is not None else np.zeros(matrix.shape[0])
            gain = readable * conditional
        return np.clip(np.column_stack([gain, readable]), 0, 1)


def combine_components(predictions, weights):
    return sum(weight * predictions[name] for name, weight in weights.items())


def policy_priorities(ids, components, fraction, policy):
    from .metric import make_priorities
    ids = np.asarray(ids, dtype=str)
    components = np.asarray(components, dtype=float)
    if components.shape != (len(ids), 2) or not np.all(np.isfinite(components)):
        raise ValueError('Expected finite Nx2 [gain, probability_read] predictions.')
    if np.any((components < 0) | (components > 1)):
        raise ValueError('Gain and read probability must lie in [0, 1].')
    gain, readable = components[:, 0], components[:, 1]
    if policy == 'gain':
        selection = gain
    elif policy == 'read':
        selection = readable
    elif policy == 'utility_half':
        selection = gain + .5 * readable
    elif policy == 'utility_one':
        selection = gain + readable
    else:
        raise ValueError(f'Unknown policy: {policy}')
    chosen = make_priorities(ids, selection, fraction) > 0
    result = np.zeros(len(ids))
    result[chosen] = make_priorities(ids[chosen], gain[chosen], 1.0)
    return result
