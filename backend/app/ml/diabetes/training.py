"""Baseline candidate training for the diabetes readmission task."""

from __future__ import annotations

from time import perf_counter
from typing import NamedTuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from backend.app.ml.diabetes.preprocessing import build_diabetes_preprocessor


class CandidateTrainingRun(NamedTuple):
    models: dict[str, Pipeline]
    fit_seconds: dict[str, float]


def build_candidate_estimators(
    training_target: pd.Series,
    *,
    random_state: int = 42,
) -> dict[str, object]:
    """Create untuned baseline estimators using class weighting for imbalance."""
    class_counts = training_target.value_counts()
    if set(class_counts.index) != {0, 1}:
        raise ValueError("Training target must contain both binary classes, 0 and 1")

    negative_count = int(class_counts.loc[0])
    positive_count = int(class_counts.loc[1])
    positive_weight = negative_count / positive_count

    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_estimators=100,
            n_jobs=-1,
            random_state=random_state,
        ),
        "linear_svm": LinearSVC(
            class_weight="balanced",
            dual="auto",
            max_iter=5000,
            random_state=random_state,
        ),
        "xgboost": XGBClassifier(
            colsample_bytree=0.85,
            eval_metric="logloss",
            learning_rate=0.08,
            max_depth=5,
            n_estimators=120,
            n_jobs=-1,
            objective="binary:logistic",
            random_state=random_state,
            reg_lambda=1.0,
            scale_pos_weight=positive_weight,
            subsample=0.85,
            tree_method="hist",
        ),
    }


def train_candidate_models(
    training_features: pd.DataFrame,
    training_target: pd.Series,
    *,
    random_state: int = 42,
) -> CandidateTrainingRun:
    """Fit each untuned candidate on training data only and report fit times.

    Every candidate receives its own sklearn pipeline, so imputation, scaling,
    and category encoding are learned only from the supplied training partition.
    This function neither evaluates the holdout nor serializes fitted models.
    """
    if not training_features.index.equals(training_target.index):
        raise ValueError("Training features and target must have aligned indexes")
    if training_features.empty:
        raise ValueError("Training data must not be empty")

    estimators = build_candidate_estimators(
        training_target,
        random_state=random_state,
    )
    fitted_models: dict[str, Pipeline] = {}
    fit_seconds: dict[str, float] = {}

    for name, estimator in estimators.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_diabetes_preprocessor(training_features)),
                ("classifier", estimator),
            ]
        )
        start_time = perf_counter()
        pipeline.fit(training_features, training_target)
        fit_seconds[name] = perf_counter() - start_time
        fitted_models[name] = pipeline

    return CandidateTrainingRun(fitted_models, fit_seconds)