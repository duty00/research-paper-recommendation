"""CPU models: train-only text vocabularies, categorical metadata and numeric features."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

BASE_MODELS = {
    "ridge_a1": {"family": "regression", "alpha": 1.0},
    "ridge_a10": {"family": "regression", "alpha": 10.0},
    "logistic_c1": {"family": "classification", "C": 1.0},
    "logistic_c4": {"family": "classification", "C": 4.0},
}
# Blends are declared before validation, not chosen after seeing holdout results.
CANDIDATES = {name: {"family": spec["family"], "weights": {name: 1.0}}
              for name, spec in BASE_MODELS.items()}
for weight in (0.25, 0.5, 0.75):
    CANDIDATES[f"blend_ridge10_logistic1_w{int(100 * weight)}"] = {
        "family": "blend", "weights": {"ridge_a10": weight, "logistic_c1": 1 - weight}}


def rows_to_x(rows: list[dict]) -> np.ndarray:
    # Identifiers and DOI/URL text are deliberately excluded from predictors.
    return np.asarray([
        [row["title"], row["venue"] or "_missing_", row["authors"] or "_missing_",
         (float(row["year"]) - 2020) / 10 if row["year"] else 0,
         float(not row["year"]), np.log1p(len(row["title"].split())) / 3,
         float(bool(row["authors"]))]
        for row in rows], dtype=object)


def as_float(values):
    return np.asarray(values, dtype=np.float64)


def build_model(name: str, seed: int) -> Pipeline:
    features = ColumnTransformer(
        [
            ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=18000,
                                      sublinear_tf=True, strip_accents="unicode"), 0),
            ("characters", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                           min_df=2, max_features=30000, sublinear_tf=True), 0),
            ("venue", OneHotEncoder(handle_unknown="ignore", dtype=np.float64), [1]),
            ("authors", TfidfVectorizer(min_df=2, max_features=6000, strip_accents="unicode"), 2),
            ("numeric", FunctionTransformer(as_float), [3, 4, 5, 6]),
        ],
        transformer_weights={"words": 1.0, "characters": 0.6, "venue": 0.4,
                             "authors": 0.2, "numeric": 0.1},
        sparse_threshold=1.0,
    )
    spec = BASE_MODELS[name]
    if spec["family"] == "regression":
        estimator = Ridge(alpha=spec["alpha"], solver="lsqr", tol=1e-6)
    else:
        estimator = LogisticRegression(C=spec["C"], solver="lbfgs", max_iter=1500,
                                       tol=1e-5, random_state=seed)
    return Pipeline([("features", features), ("estimator", estimator)])


def fit_model(name, x, y, seed):
    model = build_model(name, seed)
    target = (np.asarray(y) - 1) / 4 if BASE_MODELS[name]["family"] == "regression" else y
    return model.fit(x, target)


def predict_gain(model, x):
    if hasattr(model, "predict_proba"):
        gain = model.predict_proba(x) @ ((model.classes_.astype(float) - 1) / 4)
    else:
        gain = model.predict(x)
    return np.clip(np.asarray(gain), 0, 1)


def combine(predictions: dict, weights: dict) -> np.ndarray:
    return sum(weight * predictions[name] for name, weight in weights.items())
