"""Supervised heads on frozen embeddings; centering/metadata fit inside folds."""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.svm import SVC
from .advanced import fit_binary, positive_probability

HEADS = {
    'ordinal_c4': dict(kind='ordinal', C=4., metadata=False),
    'ordinal_c16': dict(kind='ordinal', C=16., metadata=False),
    'ordinal_meta_c4': dict(kind='ordinal', C=4., metadata=True),
    'multi_c4': dict(kind='multiclass', C=4., metadata=False),
    'svc_c4': dict(kind='svc', C=4., metadata=False),
    'ridge_a1': dict(kind='ridge', alpha=1., metadata=False),
}
SEMANTIC_SPECS = {f'{encoder}_{head}': dict(encoder=encoder, **spec)
                  for encoder in ('minilm', 'e5') for head, spec in HEADS.items()}
BASELINES = {'v2_title': dict(profile='title', kind='ordinal', C=1.),
             'v2_rich': dict(profile='rich', kind='ordinal', C=2.)}
WEIGHTS = {name: {name: 1.} for name in [*SEMANTIC_SPECS, *BASELINES]}
WEIGHTS.update({f'blend_{name}_title': {name: .5, 'v2_title': .5} for name in SEMANTIC_SPECS})
WEIGHTS['blend_encoders_ordinal'] = {'minilm_ordinal_c4': .5, 'e5_ordinal_c4': .5}


class SemanticHead(BaseEstimator):
    def __init__(self, spec, seed=261):
        self.spec = spec
        self.seed = seed

    def transform(self, embeddings, x):
        values = normalize(np.asarray(embeddings, dtype=float) - self.center_)
        if self.spec['metadata']:
            venue = self.venue_.transform(x[:, [1]]) * .35
            numeric = np.asarray(x[:, [3, 4, 5, 6]], dtype=float) * .05
            values = np.column_stack([values, venue, numeric])
        return values

    def fit(self, embeddings, x, y):
        self.center_ = np.mean(np.asarray(embeddings, dtype=float), axis=0)
        if self.spec['metadata']:
            self.venue_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(x[:, [1]])
        features = self.transform(embeddings, x)
        y = np.asarray(y, dtype=int)
        kind = self.spec['kind']
        if kind == 'ordinal':
            self.models_ = [fit_binary(features, (y > boundary).astype(int), self.spec['C'], self.seed)
                            for boundary in (1, 2, 3, 4)]
        elif kind in ('multiclass', 'svc'):
            if kind == 'svc':
                self.model_ = SVC(C=self.spec['C'], probability=True, random_state=self.seed)
            else:
                self.model_ = LogisticRegression(C=self.spec['C'], max_iter=2000, tol=1e-5, random_state=self.seed)
            self.model_.fit(features, y)
        elif kind == 'ridge':
            self.model_ = Ridge(alpha=self.spec['alpha']).fit(features,
                            np.column_stack([(y - 1) / 4, (y > 1).astype(float)]))
        else:
            raise ValueError(f'Unknown semantic head: {kind}')
        return self

    def predict_components(self, embeddings, x):
        features = self.transform(embeddings, x)
        kind = self.spec['kind']
        if kind == 'ordinal':
            thresholds = np.column_stack([positive_probability(model, features) for model in self.models_])
            result = np.column_stack([thresholds.mean(axis=1), thresholds[:, 0]])
        elif kind in ('multiclass', 'svc'):
            prob = self.model_.predict_proba(features)
            classes = list(self.model_.classes_)
            gain = prob @ ((np.asarray(classes) - 1) / 4)
            read = 1 - prob[:, classes.index(1)] if 1 in classes else np.ones(len(features))
            result = np.column_stack([gain, read])
        else:
            result = self.model_.predict(features)
        return np.clip(result, 0, 1)
